"""
make_jsonl_dataset.py
---------------------
Generate Sokoban Train/Val/Test JSONL datasets directly (no Mongo required).

- Train: in-domain (base) only
- Val/Test: mix of in-domain (base) and OOD (harder) by ratios
- Uses your existing `sokoban.generate_level` and A* to build traces

Example:
  python make_jsonl_dataset.py \
    --out ./data/soko_jsonl \
    --train 10000 --val 2000 --test 2000 \
    --base "7x7:walls=3,boxes=2" \
    --ood  "10x10:walls=5,boxes=2" \
    --val-ood 0.5 --test-ood 0.5 \
    --seed 0
"""

import json, time, random, argparse, re
from pathlib import Path
from typing import Tuple

# Flat imports assuming this file sits next to your project files
from .sokoban import Sokoban, SokobanTrace, generate_level, AStarSokobanState, astar, AStarCannotSolveTaskException
from .sokoban_difficulty import SokobanDifficultyModifier  # for measure_difficulty

def parse_spec(spec: str) -> Tuple[int,int,int,int]:
    # "7x7:walls=3,boxes=2" -> (7,7,3,2)
    size, *rest = spec.split(":", 1)
    w, h = map(int, size.lower().split("x"))
    walls, boxes = 3, 2
    if rest:
        kv = {}
        for part in rest[0].split(","):
            part = part.strip()
            if not part or "=" not in part: continue
            k,v = part.split("=",1)
            kv[k.strip()] = int(v.strip())
        walls = kv.get("walls", walls)
        boxes = kv.get("boxes", boxes)
    return w, h, walls, boxes

def ensure_paths(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"format":"sokoban-jsonl-v1","created_at":time.time()}, indent=2))
    return {
        "train": root / "train.jsonl",
        "val":   root / "val.jsonl",
        "test":  root / "test.jsonl",
    }

def write_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def gen_n(paths, split: str, n: int, w: int, h: int, walls: int, boxes: int, domain: str, measurer: SokobanDifficultyModifier):
    done = 0
    while done < n:
        try:
            s = generate_level(w, h, walls, boxes)
            tr = list(astar(AStarSokobanState(s, deterministic=False)))
            st = SokobanTrace(s, tr)
            _id = abs(hash(f"{hash(st)}_{time.time_ns()}")) % (10**12)
            metrics = measurer.measure_difficulty(s)
            obj = {
                "_id": _id,
                "parent_id": None,
                "split": split,
                "domain": domain,
                "spec": {"width": w, "height": h, "num_walls": walls, "num_boxes": boxes},
                "metrics": metrics,
                "trace": st.to_dict(),
            }
            write_jsonl(paths[split], obj)
            done += 1
        except AStarCannotSolveTaskException:
            continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, required=True)
    ap.add_argument("--val", type=int, required=True)
    ap.add_argument("--test", type=int, required=True)
    ap.add_argument("--base", default="7x7:walls=3,boxes=2")
    ap.add_argument("--ood",  default="10x10:walls=5,boxes=2")
    ap.add_argument("--val-ood", type=float, default=0.5)
    ap.add_argument("--test-ood", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    paths = ensure_paths(out)

    b_w, b_h, b_walls, b_boxes = parse_spec(args.base)
    o_w, o_h, o_walls, o_boxes = parse_spec(args.ood)

    # Use the difficulty modifier just for measuring difficulty (no Mongo write)
    measurer = SokobanDifficultyModifier(f"sokoban.{b_w}-by-{b_h}-walls-{b_walls}-boxes-{b_boxes}", create_modified_dataset=False, out_dir=None)

    # Train: base only (in-domain)
    print(f"[train] {args.train} × base({b_w}x{b_h}, walls={b_walls}, boxes={b_boxes})")
    gen_n(paths, "train", args.train, b_w, b_h, b_walls, b_boxes, "in", measurer)

    # Val/Test mixtures
    n_val_out = int(round(args.val * args.val_ood))
    n_val_in  = args.val - n_val_out
    n_test_out = int(round(args.test * args.test_ood))
    n_test_in  = args.test - n_test_out

    print(f"[val.in]  {n_val_in} × base({b_w}x{b_h})")
    gen_n(paths, "val", n_val_in, b_w, b_h, b_walls, b_boxes, "in", measurer)
    print(f"[val.out] {n_val_out} × ood({o_w}x{o_h})")
    # new measurer for OOD dims (for correct tokenizer sizing if you add that later)
    measurer_ood = SokobanDifficultyModifier(f"sokoban.{o_w}-by-{o_h}-walls-{o_walls}-boxes-{o_boxes}", create_modified_dataset=False, out_dir=None)
    gen_n(paths, "val", n_val_out, o_w, o_h, o_walls, o_boxes, "out", measurer_ood)

    print(f"[test.in] {n_test_in} × base({b_w}x{b_h})")
    gen_n(paths, "test", n_test_in, b_w, b_h, b_walls, b_boxes, "in", measurer)
    print(f"[test.out]{n_test_out} × ood({o_w}x{o_h})")
    gen_n(paths, "test", n_test_out, o_w, o_h, o_walls, o_boxes, "out", measurer_ood)

    # Write meta for convenience
    meta = {
        "seed": args.seed,
        "base": {"width": b_w, "height": b_h, "num_walls": b_walls, "num_boxes": b_boxes},
        "ood":  {"width": o_w, "height": o_h, "num_walls": o_walls, "num_boxes": o_boxes},
        "splits": {
            "train": {"n": args.train, "domain": "in"},
            "val":   {"n": args.val,   "ood_fraction": args.val_ood},
            "test":  {"n": args.test,  "ood_fraction": args.test_ood},
        }
    }
    (Path(args.out)/"meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote JSONL dataset to: {args.out}")

if __name__ == "__main__":
    main()