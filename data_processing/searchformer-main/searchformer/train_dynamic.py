
"""
train_dynamic.py
----------------
A minimal dynamic-difficulty training scaffold.

What it does:
- Loads an initial JSONL dataset (created by export_and_push_hf.py or your own).
- Periodically *probes* a subset to see which puzzles are too easy or too hard.
- Uses heuristics from `sokoban_difficulty.py` to adjust those puzzles.
- Keeps provenance (parent_id + transform history) and always preserves originals.
- Swaps which version is currently active for training.

This file is **model-agnostic**. Plug your model by providing a `eval_batch_fn`
that maps a list of examples -> per-example scores (loss or success rate).
If you don't provide one, it falls back to the A* difficulty proxy (plan length, etc.).

Run (toy, A*-based adjustment):
  python train_dynamic.py \
    --data ./data/soko_run1 \
    --base "7x7:walls=3,boxes=2" \
    --target-level medium \
    --probe-size 256 \
    --adjust-every 5000
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import click

# Local imports (flat folder assumption)
from sokoban_difficulty import (
    JsonlDataset,
    SokobanDifficultyModifier,
    _parse_spec,
    auto_adjust_difficulty,  # tuning loop
    _level_to_band,
)
from sokoban import Sokoban, SokobanTrace
from trace import TokenizedTrace

Split = Literal["train","val","test"]
Domain = Literal["in","out"]

@dataclass
class VersionedExample:
    """A single version of an example (could be original or transformed)."""
    id: int
    parent_id: Optional[int]
    split: Split
    domain: Domain
    spec: Dict[str, Any]
    metrics: Dict[str, Any]
    transform_history: List[Dict[str, Any]]
    trace: Dict[str, Any]  # serialized SokobanTrace

    def to_json(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "parent_id": self.parent_id,
            "split": self.split,
            "domain": self.domain,
            "spec": self.spec,
            "metrics": self.metrics,
            "transform_history": self.transform_history,
            "trace": self.trace,
        }

class DynamicPool:
    """
    Maintains an active training pool with per-example versions.
    - `active_version[root_id] -> current_id`
    - All versions are stored in the same JSONL (append-only).
    """
    def __init__(self, data_dir: str, base_spec: str, seed: int=0):
        random.seed(seed)
        self.store = JsonlDataset(data_dir)
        self.base_spec = base_spec
        self.versions_index_path = Path(self.store.paths.root) / "versions.index.json"

        self.active_version: Dict[int, int] = {}  # root_id -> current_id
        self.root_to_children: Dict[int, List[int]] = {}
        self.id_to_doc: Dict[int, Dict[str, Any]] = {}

        self._load_all()

    def _load_all(self):
        for split in ["train","val","test"]:
            for obj in self.store.iter_split(split):
                root_id = obj.get("parent_id", obj["_id"])
                self.id_to_doc[obj["_id"]] = obj
                self.active_version.setdefault(root_id, obj["_id"])
                self.root_to_children.setdefault(root_id, []).append(obj["_id"])
        # Load overrides if any
        if self.versions_index_path.exists():
            override = json.loads(self.versions_index_path.read_text())
            for k,v in override.items():
                self.active_version[int(k)] = int(v)

    def _save_index(self):
        tmp = {str(k): int(v) for k,v in self.active_version.items()}
        self.versions_index_path.write_text(json.dumps(tmp, indent=2))

    def sample_roots(self, k: int) -> List[int]:
        roots = list(self.active_version.keys())
        random.shuffle(roots)
        return roots[:k]

    def get_current_examples(self, root_ids: List[int]) -> List[VersionedExample]:
        out: List[VersionedExample] = []
        for rid in root_ids:
            vid = self.active_version[rid]
            doc = self.id_to_doc[vid]
            out.append(VersionedExample(
                id=doc["_id"],
                parent_id=doc.get("parent_id"),
                split=doc.get("split","train"),
                domain=doc.get("domain","in"),
                spec=doc.get("spec",{}),
                metrics=doc.get("metrics",{}),
                transform_history=doc.get("transform_history",[]),
                trace=doc["trace"],
            ))
        return out

    def swap_active(self, root_id: int, new_id: int):
        self.active_version[root_id] = new_id
        self._save_index()

    def add_version(self, split: Split, domain: Domain, parent_id: int, trace_obj: SokobanTrace, metrics: Dict[str, Any], history: List[Dict[str, Any]]) -> int:
        # store like JsonlDataset.add_trace but with extra fields
        new_id = abs(hash(f"{hash(trace_obj)}_{time.time_ns()}")) % (10**12)
        obj = {
            "_id": new_id,
            "parent_id": parent_id,
            "split": split,
            "domain": domain,
            "spec": {
                "width": trace_obj.sokoban_start.width,
                "height": trace_obj.sokoban_start.height,
                "num_walls": None,
                "num_boxes": None,
            },
            "metrics": metrics,
            "transform_history": history,
            "trace": trace_obj.to_dict(),
        }
        self.store.add_trace(split, obj)
        self.id_to_doc[new_id] = obj
        self.root_to_children.setdefault(parent_id, []).append(new_id)
        return new_id


def default_probe_score(ex: VersionedExample) -> float:
    """Fallback probe = A* difficulty score (no model needed)."""
    # If metrics already present, use it. Otherwise compute proxy from trace length.
    metrics = ex.metrics or {}
    score = metrics.get("difficulty_score")
    if score is not None and not math.isinf(score):
        return float(score)
    # Approx: solution length as a proxy if metrics not present
    trace = SokobanTrace.from_dict(ex.trace)
    plan_len = sum(1 for s in trace.trace if s["action"] == "plan")
    return float(plan_len)


def adjust_once(
    pool: DynamicPool,
    mod: SokobanDifficultyModifier,
    probe_size: int,
    target_band: Tuple[float,float],
    eval_batch_fn: Optional[Any] = None,
    split: Split = "train",
    domain: Domain = "in",
) -> Dict[str, Any]:
    """
    Probe a subset, mark too-easy / too-hard, and adjust by one heuristic step.
    Returns simple stats about how many were adjusted.
    """
    root_ids = pool.sample_roots(probe_size)
    examples = pool.get_current_examples(root_ids)

    # 1) Score each example
    if eval_batch_fn is None:
        scores = [default_probe_score(ex) for ex in examples]
    else:
        scores = eval_batch_fn(examples)  # user-provided list[float] (lower=easier)
        assert len(scores) == len(examples)

    # 2) Partition
    too_easy, in_band, too_hard = [], [], []
    lo, hi = target_band
    for ex, s in zip(examples, scores):
        if s < lo:       too_easy.append(ex)
        elif s >= hi:    too_hard.append(ex)
        else:            in_band.append(ex)

    adjusted_easy = adjusted_hard = 0
    # 3) Adjust and swap active version if solvable & closer to band
    for ex in too_easy:
        soko = SokobanTrace.from_dict(ex.trace).sokoban_start
        tuned, info = auto_adjust_difficulty(mod, soko, target_band, max_steps=1)  # one nudge harder
        metrics = mod.measure_difficulty(tuned)
        if metrics["is_solvable"]:
            trace = SokobanTrace(tuned, list([]))  # Re-solve to get full trace
            try:
                from sokoban import astar, AStarSokobanState
                trace = SokobanTrace(tuned, list(astar(AStarSokobanState(tuned, deterministic=False))))
            except Exception:
                pass
            new_id = pool.add_version(split, domain, parent_id=ex.parent_id or ex.id, trace_obj=trace, metrics=metrics, history=info.get("history",[]))
            pool.swap_active(ex.parent_id or ex.id, new_id)
            adjusted_easy += 1

    for ex in too_hard:
        soko = SokobanTrace.from_dict(ex.trace).sokoban_start
        # Ask the tuner to move toward band; it will choose an easier op first
        tuned, info = auto_adjust_difficulty(mod, soko, target_band, max_steps=1)
        metrics = mod.measure_difficulty(tuned)
        if metrics["is_solvable"]:
            trace = SokobanTrace(tuned, list([]))
            try:
                from sokoban import astar, AStarSokobanState
                trace = SokobanTrace(tuned, list(astar(AStarSokobanState(tuned, deterministic=False))))
            except Exception:
                pass
            new_id = pool.add_version(split, domain, parent_id=ex.parent_id or ex.id, trace_obj=trace, metrics=metrics, history=info.get("history",[]))
            pool.swap_active(ex.parent_id or ex.id, new_id)
            adjusted_hard += 1

    return {
        "probed": len(examples),
        "too_easy": len(too_easy),
        "too_hard": len(too_hard),
        "in_band": len(in_band),
        "adjusted_easy": adjusted_easy,
        "adjusted_hard": adjusted_hard,
    }


@click.command()
@click.option("--data", "data_dir", type=click.Path(file_okay=False), required=True, help="Directory with train/val/test JSONL.")
@click.option("--base", "base_spec", type=str, default="7x7:walls=3,boxes=2")
@click.option("--target-level", type=click.Choice(["easy","medium","hard"]), default="medium")
@click.option("--probe-size", type=int, default=256, help="How many roots to probe per adjustment pass.")
@click.option("--adjust-every", type=int, default=5000, help="Call adjust_once() every N training examples.")
@click.option("--seed", type=int, default=0)
def main(data_dir: str, base_spec: str, target_level: str, probe_size: int, adjust_every: int, seed: int):
    random.seed(seed)

    # Band
    band = _level_to_band(target_level)

    # Pool + modifier (file-backed)
    mod = SokobanDifficultyModifier("sokoban.dynamic", out_dir=data_dir, tokenize=False)
    pool = DynamicPool(data_dir, base_spec, seed=seed)

    # ----- PSEUDO TRAINING LOOP (replace with your own) -----
    seen = 0
    epoch = 0
    while True:
        # Your training step goes here; e.g., iterate batches from your loader
        # For demo, we just "simulate" training by sleeping a bit
        time.sleep(0.5)
        seen += 512  # pretend we trained on 512 examples

        if seen >= adjust_every:
            seen = 0
            stats = adjust_once(pool, mod, probe_size=probe_size, target_band=band, eval_batch_fn=None, split="train", domain="in")
            click.echo(f"[adjust] epoch={epoch} {stats}")
            epoch += 1
        if epoch >= 5:
            break

if __name__ == "__main__":
    main()
