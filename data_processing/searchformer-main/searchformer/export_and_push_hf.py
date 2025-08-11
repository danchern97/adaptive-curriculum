
"""
export_and_push_hf.py
---------------------
Generate Sokoban datasets (Train/Val/Test) using the original Searchformer
generation flow, write JSONL files to disk, and (optionally) push to
Hugging Face Datasets.

This script uses helpers from `sokoban_difficulty.py` but DOES NOT require
MongoDB; everything is file-first. You can still run the legacy Mongo pipelines
elsewhere if you want.

Usage (example):
  python export_and_push_hf.py \
    --out ./data/soko_run1 \
    --train 10000 --val 2000 --test 2000 \
    --base "7x7:walls=3,boxes=2" \
    --ood  "10x10:walls=5,boxes=2" \
    --val-ood 0.5 --test-ood 0.5 \
    --seed 0 \
    --repo-id yourname/sokoban-srun1 --private

If you don't pass --repo-id, it will just write local JSONL files.
"""

import json
import os
import sys
import time
import logging
import random
from pathlib import Path
from typing import Optional

import click

# Local imports (flat folder assumption; change to package-style if needed)
from sokoban_difficulty import (
    JsonlDataset,
    _parse_spec,
    _generate_and_store,
)

# Optional libs for pushing to HF
def _lazy_import_datasets():
    try:
        import datasets  # type: ignore
        from datasets import Dataset, DatasetDict  # type: ignore
        return datasets, Dataset, DatasetDict
    except Exception as e:
        raise RuntimeError("`datasets` library not available. `pip install datasets`") from e

def _lazy_import_hf_hub():
    try:
        from huggingface_hub import HfApi, create_repo  # type: ignore
        return HfApi, create_repo
    except Exception as e:
        raise RuntimeError("`huggingface_hub` not available. `pip install huggingface_hub`") from e


@click.command()
@click.option("--out", "out_dir", type=click.Path(file_okay=False), required=True, help="Output directory for JSONL files.")
@click.option("--train", "n_train", type=int, required=True, help="Number of in-domain training examples (base spec).")
@click.option("--val", "n_val", type=int, required=True, help="Number of validation examples (mixture).")
@click.option("--test", "n_test", type=int, required=True, help="Number of test examples (mixture).")
@click.option("--base", "base_spec", type=str, default="7x7:walls=3,boxes=2", help='In-domain spec, e.g. "7x7:walls=3,boxes=2"')
@click.option("--ood",  "ood_spec",  type=str, default="10x10:walls=5,boxes=2", help='OOD spec, e.g. "10x10:walls=5,boxes=2"')
@click.option("--val-ood", type=float, default=0.5, help="Fraction of OOD in validation (0..1).")
@click.option("--test-ood", type=float, default=0.5, help="Fraction of OOD in test (0..1).")
@click.option("--tokenize", is_flag=True, help="Also emit tokenized JSONL files using WithBoxSokobanTokenizer.")
@click.option("--seed", type=int, default=0, help="RNG seed for reproducibility.")
@click.option("--repo-id", type=str, default=None, help="Hugging Face repo id, e.g. 'user/sokoban-7x7-ood'. If omitted, skip push.")
@click.option("--private", is_flag=True, help="Create/push as a private dataset on HF.")
@click.option("--commit-msg", type=str, default="initial dataset push")
def main(
    out_dir: str,
    n_train: int,
    n_val: int,
    n_test: int,
    base_spec: str,
    ood_spec: str,
    val_ood: float,
    test_ood: float,
    tokenize: bool,
    seed: int,
    repo_id: Optional[str],
    private: bool,
    commit_msg: str,
):
    random.seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    store = JsonlDataset(out)

    b_w, b_h, b_walls, b_boxes = _parse_spec(base_spec)
    o_w, o_h, o_walls, o_boxes = _parse_spec(ood_spec)

    # Train: in-domain only
    click.echo(f"[export] train: {n_train} × base({b_w}x{b_h}, walls={b_walls}, boxes={b_boxes})")
    _generate_and_store(store, "train", n_train, b_w, b_h, b_walls, b_boxes, "in", tokenizer=None if not tokenize else None)  # tokenizer handled inside if passed

    # Val/Test: mixtures
    n_val_out = int(round(n_val * val_ood))
    n_val_in  = n_val - n_val_out
    n_test_out = int(round(n_test * test_ood))
    n_test_in  = n_test - n_test_out

    click.echo(f"[export] val.in:  {n_val_in}  × base({b_w}x{b_h})")
    _generate_and_store(store, "val", n_val_in, b_w, b_h, b_walls, b_boxes, "in", tokenizer=None)
    click.echo(f"[export] val.out: {n_val_out} × ood({o_w}x{o_h})")
    _generate_and_store(store, "val", n_val_out, o_w, o_h, o_walls, o_boxes, "out", tokenizer=None)

    click.echo(f"[export] test.in: {n_test_in} × base({b_w}x{b_h})")
    _generate_and_store(store, "test", n_test_in, b_w, b_h, b_walls, b_boxes, "in", tokenizer=None)
    click.echo(f"[export] test.out:{n_test_out} × ood({o_w}x{o_h})")
    _generate_and_store(store, "test", n_test_out, o_w, o_h, o_walls, o_boxes, "out", tokenizer=None)

    store.update_meta(
        seed=seed,
        base={"width": b_w, "height": b_h, "num_walls": b_walls, "num_boxes": b_boxes},
        ood={"width": o_w, "height": o_h, "num_walls": o_walls, "num_boxes": o_boxes},
        splits={
            "train": {"n": n_train, "domain": "in"},
            "val": {"n": n_val, "ood_fraction": val_ood},
            "test": {"n": n_test, "ood_fraction": test_ood},
        },
        tokenized=tokenize,
    )

    if not repo_id:
        click.echo(f"[export] wrote JSONL dataset to: {out_dir} (skipped HF push)")
        return

    # Push to HF
    datasets, Dataset, DatasetDict = _lazy_import_datasets()
    HfApi, create_repo = _lazy_import_hf_hub()

    api = HfApi()
    try:
        create_repo(repo_id, private=private, exist_ok=True)
    except Exception:
        pass

    # Read JSONL into a DatasetDict
    splits = {}
    for split_name in ["train", "val", "test"]:
        path = (store.paths.train if split_name=="train" else store.paths.val if split_name=="val" else store.paths.test)
        if path.exists():
            ds = Dataset.from_json(str(path))
            splits[split_name] = ds
    if not splits:
        raise RuntimeError("No JSONL files found to push. Did generation succeed?")

    dsd = DatasetDict(splits)
    dsd.push_to_hub(repo_id, private=private, commit_message=commit_msg)
    click.echo(f"[push] Pushed to https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
