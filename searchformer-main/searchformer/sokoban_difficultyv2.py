import hashlib
import logging
import os
import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import click
import pymongo

from .astar import AStarCannotSolveTaskException, TraceStep, astar
from .sokoban import (
    CellState,
    Sokoban,
    SokobanRenderer,
    SokobanTrace,
    SokobanTraceDataset,
    sokoban_state_to_pretty_string,
    WithBoxSokobanTokenizer,
)
from .trace import CannotTokenizeException, TokenizedDataset, TokenizedTrace
from .utils import mongodb_client
from .sokoban import SOKOBAN_DB_NAME  # constants live in sokoban.py

# ---------------------------
# Small helpers
# ---------------------------

def _history_coll(dataset: str):
    return mongodb_client()[SOKOBAN_DB_NAME][f"{dataset}.history"]

def _get_live_trace(dataset: str, seq_id: int) -> Tuple[SokobanTrace, bool]:
    ds = SokobanTraceDataset(dataset)
    d = ds.trace_collection.find_one({"_id": seq_id})
    if not d:
        raise ValueError(f"[{dataset}] _id={seq_id} not found.")
    return SokobanTrace.from_dict(d["trace"]), bool(d.get("is_test", False))

def _ascii(s: Sokoban) -> str:
    return sokoban_state_to_pretty_string(s.state)

def _plan_steps(trace: SokobanTrace) -> List[TraceStep]:
    return [t for t in trace.trace if t.action == "plan"]

def _plan_len(trace: SokobanTrace) -> int:
    return len(_plan_steps(trace))

def _grid_hash(s: Sokoban) -> str:
    return hashlib.sha1(_ascii(s).encode("utf-8")).hexdigest()

def _trace_hash(t: SokobanTrace) -> str:
    return hashlib.sha1(str(t.to_dict()).encode("utf-8")).hexdigest()

def _latest_version(hc, seq_id: int) -> Optional[int]:
    d = hc.find_one({"seq_id": seq_id}, sort=[("version", pymongo.DESCENDING)], projection={"version": 1})
    return None if d is None else int(d["version"])

def _rerun_astar_from(s: Sokoban) -> SokobanTrace:
    from .sokoban import AStarSokobanState
    return SokobanTrace(s, list(astar(AStarSokobanState(s, deterministic=False))))

def _solvable(s: Sokoban) -> bool:
    try:
        _ = _rerun_astar_from(s)
        return True
    except AStarCannotSolveTaskException:
        return False

# ---------------------------
# Minimal heuristics
# ---------------------------

def _interior(s: Sokoban):
    for y in range(1, s.height - 1):
        for x in range(1, s.width - 1):
            yield x, y

def h_partial_solve(orig_trace: SokobanTrace, pct: float) -> Tuple[Sokoban, bool]:
    plan = _plan_steps(orig_trace)
    if len(plan) <= 2:
        return orig_trace.sokoban_start, False
    idx = max(1, int(pct * (len(plan) - 1)))
    s = Sokoban(plan[idx].state["state"])
    return s, _solvable(s)

def h_remove_walls(s: Sokoban, count: int) -> Tuple[Sokoban, bool]:
    mod = Sokoban(deepcopy(s.state))
    cands = [(x, y) for x, y in _interior(mod) if mod.get_state(x, y) == CellState.wall]
    random.shuffle(cands)
    removed = 0
    for x, y in cands:
        if removed >= count:
            break
        prev = mod.get_state(x, y)
        mod.set_state(x, y, CellState.floor)
        if _solvable(mod):
            removed += 1
        else:
            mod.set_state(x, y, prev)
    return (mod if removed > 0 else s), removed > 0

def h_add_walls(s: Sokoban, count: int) -> Tuple[Sokoban, bool]:
    mod = Sokoban(deepcopy(s.state))
    cands = [(x, y) for x, y in _interior(mod) if mod.get_state(x, y) == CellState.floor]
    random.shuffle(cands)
    added = 0
    for x, y in cands:
        if added >= count:
            break
        prev = mod.get_state(x, y)
        mod.set_state(x, y, CellState.wall)
        if _solvable(mod):
            added += 1
        else:
            mod.set_state(x, y, prev)
    return (mod if added > 0 else s), added > 0

def h_add_box_and_dock(s: Sokoban) -> Tuple[Sokoban, bool]:
    mod = Sokoban(deepcopy(s.state))
    floors = [(x, y) for x, y in _interior(mod) if mod.get_state(x, y) == CellState.floor]
    random.shuffle(floors)
    for i in range(min(50, len(floors)//2)):
        bx, by = floors[i]
        dx, dy = floors[-(i+1)]
        pb, pd = mod.get_state(bx, by), mod.get_state(dx, dy)
        mod.set_state(bx, by, CellState.box)
        mod.set_state(dx, dy, CellState.dock)
        if _solvable(mod):
            return mod, True
        mod.set_state(bx, by, pb); mod.set_state(dx, dy, pd)
    return s, False

# ---------------------------
# Tokenization helpers
# ---------------------------

def _require_tok_dataset(name: str, tokenizer: WithBoxSokobanTokenizer) -> TokenizedDataset:
    td = TokenizedDataset(name)
    if not td.exists():
        raise RuntimeError(
            f"Token dataset '{name}' not found. Create it (with the right vocabulary) before mutating."
        )
    return td

def _check_vocab(td: TokenizedDataset, tok: TokenizedTrace) -> Optional[Set[str]]:
    vocab = set(td.vocabulary)
    all_tokens = set(tok.prompt) | set(tok.reasoning) | set(tok.plan)
    missing = all_tokens - vocab
    return missing if missing else None

def _upsert_tok(td: TokenizedDataset, tok: TokenizedTrace, is_test: bool):
    coll = td.test_seq_collection if is_test else td.train_seq_collection
    meta = td.test_meta_collection if is_test else td.train_meta_collection
    coll.replace_one({"_id": tok.id}, tok.to_dict(), upsert=True)
    meta.replace_one({"_id": tok.id}, tok.to_stats_dict(), upsert=True)

# ---------------------------
# History write (simple)
# ---------------------------

def _ensure_v0(dataset: str, seq_id: int):
    hc = _history_coll(dataset)
    if hc.count_documents({"seq_id": seq_id, "version": 0}, limit=1) == 0:
        live, is_test = _get_live_trace(dataset, seq_id)
        hc.insert_one({
            "_id": f"{seq_id}:0",
            "seq_id": seq_id,
            "version": 0,
            "parent_version": None,
            "heuristic": {"name": "baseline", "params": {}, "seed": None, "tag": None},
            "plan_len": _plan_len(live),
            "timestamp": time.time(),
            "tokenizable": True,  # baseline assumed stored already
            "grid_hash": _grid_hash(live.sokoban_start),
            "trace_hash": _trace_hash(live),
            "trace": live.to_dict(),
            "is_test": is_test,
        })

def _append_version(dataset: str, seq_id: int, parent_v: int, trace: SokobanTrace,
                    is_test: bool, heuristic: Dict[str, Any], tokenizable: bool) -> int:
    hc = _history_coll(dataset)
    latest = _latest_version(hc, seq_id)
    v_next = 1 if latest is None else latest + 1
    hc.insert_one({
        "_id": f"{seq_id}:{v_next}",
        "seq_id": seq_id,
        "version": v_next,
        "parent_version": parent_v,
        "heuristic": heuristic,
        "plan_len": _plan_len(trace),
        "timestamp": time.time(),
        "tokenizable": tokenizable,
        "grid_hash": _grid_hash(trace.sokoban_start),
        "trace_hash": _trace_hash(trace),
        "trace": trace.to_dict(),
        "is_test": is_test,
    })
    return v_next

def _write_live(dataset_out: str, seq_id: int, trace: SokobanTrace, is_test: bool):
    ds_out = SokobanTraceDataset(dataset_out)
    ds_out.trace_collection.replace_one(
        {"_id": seq_id},
        {"_id": seq_id, "is_test": is_test, "trace": trace.to_dict()},
        upsert=True,
    )
    ds_out.index_collection.replace_one(
        {"_id": seq_id},
        {"_id": seq_id, "is_test": is_test},
        upsert=True,
    )

# ---------------------------
# CLI
# ---------------------------

@click.group()
def main():
    """Sokoban mutator with a tiny versioned history (Mongo-only)."""
    pass

@main.command("mutate")
@click.option("--dataset", required=True, type=str)
@click.option("--id", "seq_id", required=True, type=int)
@click.option("--heuristic", required=True,
              type=click.Choice(["partial_solve", "remove_walls", "add_walls", "add_box"]))
@click.option("--out", "tok_out", required=True, type=str, help="Existing token dataset to write to.")
@click.option("--trace-out", type=str, default=None, help="Write live mutated trace into this dataset (default: --dataset).")
@click.option("--percentage", type=float, default=0.3, help="For partial_solve.")
@click.option("--count", type=int, default=1, help="For add/remove walls.")
@click.option("--tag", type=str, default=None, help="Optional label in history.")
@click.option("--seed", type=int, default=None, help="Deterministic mutation.")
@click.option("--render-images", is_flag=True)
@click.option("--render-dir", type=str, default="./render")
def mutate_cmd(dataset, seq_id, heuristic, tok_out, trace_out, percentage, count, tag, seed, render_images, render_dir):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    if seed is not None:
        random.seed(seed)

    live_trace, is_test = _get_live_trace(dataset, seq_id)
    orig = live_trace.sokoban_start

    print("\n=== ORIGINAL ==="); print(_ascii(orig))

    # 1) ensure v0 exists once per id
    _ensure_v0(dataset, seq_id)

    # 2) mutate start state
    if heuristic == "partial_solve":
        new_s, ok = h_partial_solve(live_trace, pct=percentage)
        hparams = {"percentage": percentage}
    elif heuristic == "remove_walls":
        new_s, ok = h_remove_walls(orig, count=count); hparams = {"count": count}
    elif heuristic == "add_walls":
        new_s, ok = h_add_walls(orig, count=count); hparams = {"count": count}
    elif heuristic == "add_box":
        new_s, ok = h_add_box_and_dock(orig); hparams = {}
    else:
        raise ValueError("unknown heuristic")
    if not ok:
        print(f"\n[FAIL] '{heuristic}' couldn’t find a solvable mutation.")
        return

    # 3) solve mutated
    try:
        mut_trace = _rerun_astar_from(new_s)
    except AStarCannotSolveTaskException:
        print("\n[FAIL] A* could not solve mutated puzzle.")
        return

    print("\n=== MUTATED ==="); print(_ascii(new_s))

    # 4) tokenize mutated into existing token dataset
    tokenizer = WithBoxSokobanTokenizer(new_s.width, new_s.height)
    td = _require_tok_dataset(tok_out, tokenizer)
    try:
        tok = tokenizer(mut_trace, is_test=is_test)
        # align id to seq_id to match training loaders
        tok = TokenizedTrace(id=seq_id, prompt=tok.prompt, reasoning=tok.reasoning, plan=tok.plan)
        missing = _check_vocab(td, tok)
        if missing:
            print(f"\n[tokenize] FAIL: {len(missing)} missing tokens in '{tok_out}'. Example: {sorted(list(missing))[:20]} ...")
            return
        _upsert_tok(td, tok, is_test=is_test)
        tokenizable = True
    except CannotTokenizeException as e:
        print(f"\n[tokenize] FAIL: {e}")
        return

    # 5) write history version k and replace live
    hc = _history_coll(dataset)
    parent_v = _latest_version(hc, seq_id) or 0
    v = _append_version(
        dataset=dataset,
        seq_id=seq_id,
        parent_v=parent_v,
        trace=mut_trace,
        is_test=is_test,
        heuristic={"name": heuristic, "params": hparams, "seed": seed, "tag": tag},
        tokenizable=tokenizable,
    )

    dataset_out = trace_out or dataset
    _write_live(dataset_out, seq_id, mut_trace, is_test=is_test)

    # 6) optional render
    if render_images:
        os.makedirs(render_dir, exist_ok=True)
        try:
            import pygame  # noqa: F401
            pygame.init()
            r0 = SokobanRenderer(orig.width, orig.height, record_dir=render_dir); r0.render(orig.state); r0.img_to_file()
            r1 = SokobanRenderer(new_s.width, new_s.height, record_dir=render_dir); r1.render(new_s.state); r1.img_to_file()
            try: pygame.quit()
            except Exception: pass
            print(f"[render] images in: {os.path.abspath(render_dir)}")
        except Exception as e:
            print(f"[render] skipped: {e}")

    print("\n[OK] mutation stored.")
    print(f"    history {dataset}.history: version={v} (parent={parent_v})")
    print(f"    live trace: {dataset_out}  id={seq_id}")
    print(f"    tokenized:  {tok_out}")
    print(f"    plan_len:   orig={_plan_len(live_trace)}  mutated={_plan_len(mut_trace)}  Δ={_plan_len(mut_trace)-_plan_len(live_trace)}")

# ---- history tools (list/show/revert) -------------------------------------

@main.group("history")
def history_cmd():
    pass

@history_cmd.command("list")
@click.option("--dataset", required=True, type=str)
@click.option("--id", "seq_id", required=True, type=int)
def history_list(dataset: str, seq_id: int):
    hc = _history_coll(dataset)
    cur = hc.find({"seq_id": seq_id}, sort=[("version", pymongo.ASCENDING)])
    print(f"Versions for {dataset}:{seq_id}")
    for d in cur:
        meta = d.get("heuristic") or {}
        print(f"  v={d['version']:>3}  plan_len={d.get('plan_len')}  tokenizable={d.get('tokenizable')}  "
              f"{meta.get('name','baseline')}{' tag='+meta['tag'] if meta.get('tag') else ''}  t={int(d.get('timestamp',0))}")

@history_cmd.command("show")
@click.option("--dataset", required=True, type=str)
@click.option("--id", "seq_id", required=True, type=int)
@click.option("--version", required=True, type=int)
def history_show(dataset: str, seq_id: int, version: int):
    d = _history_coll(dataset).find_one({"_id": f"{seq_id}:{version}"})
    if not d:
        print("Not found.")
        return
    tr = SokobanTrace.from_dict(d["trace"])
    print(f"{dataset}:{seq_id} v{version}  plan_len={d.get('plan_len')} tokenizable={d.get('tokenizable')}")
    print(_ascii(tr.sokoban_start))

@history_cmd.command("revert")
@click.option("--dataset", required=True, type=str)
@click.option("--id", "seq_id", required=True, type=int)
@click.option("--version", type=int, default=0, help="Version to revert to (default v0).")
@click.option("--trace-out", type=str, default=None, help="Write reverted live into this dataset (default: --dataset).")
@click.option("--out", "tok_out", type=str, default=None, help="Optional: re-tokenize reverted trace into this token dataset.")
def history_revert(dataset: str, seq_id: int, version: int, trace_out: Optional[str], tok_out: Optional[str]):
    d = _history_coll(dataset).find_one({"_id": f"{seq_id}:{version}"})
    if not d:
        print("Not found.")
        return
    tr = SokobanTrace.from_dict(d["trace"])
    is_test = bool(d.get("is_test", False))

    # write live
    dataset_out = trace_out or dataset
    _write_live(dataset_out, seq_id, tr, is_test=is_test)

    # optional retokenize
    if tok_out:
        tokenizer = WithBoxSokobanTokenizer(tr.sokoban_start.width, tr.sokoban_start.height)
        td = _require_tok_dataset(tok_out, tokenizer)
        tok = tokenizer(tr, is_test=is_test)
        tok = TokenizedTrace(id=seq_id, prompt=tok.prompt, reasoning=tok.reasoning, plan=tok.plan)
        missing = _check_vocab(td, tok)
        if missing:
            print(f"[revert] tokenization FAIL for '{tok_out}': {len(missing)} missing tokens.")
        else:
            _upsert_tok(td, tok, is_test=is_test)
            print(f"[revert] tokenized into {tok_out}")

    print(f"[revert] live set to v{version} in {dataset_out} for id={seq_id}")

if __name__ == "__main__":
    main()
