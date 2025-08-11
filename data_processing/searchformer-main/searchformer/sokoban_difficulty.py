"""
Sokoban Difficulty Modifier Module (Mongo + optional JSONL)

This module provides functionality to modify the difficulty of existing Sokoban
puzzles from an existing dataset, measure the difficulty changes, and store
modified puzzles back into MongoDB for training — OR, if `out_dir` is provided,
append them to local JSONL files instead.

Usage (Mongo unchanged):
    from searchformer.sokoban_difficulty import SokobanDifficultyModifier
    modifier = SokobanDifficultyModifier("sokoban.7-by-7-walls-2-boxes-2")
    pid = modifier.get_random_puzzle_id()
    p = modifier.get_puzzle(pid)
    p_easy, ok = modifier.make_easier(p, method="partial_solve", percentage=0.3)
    if ok:
        new_id = modifier.store_modified_puzzle(p_easy, original_id=pid, is_test=False)

Usage (JSONL file backend):
    modifier = SokobanDifficultyModifier(
        "sokoban.7-by-7-walls-2-boxes-2",
        out_dir="./data/jsonl_out"   # enables JSONL writing
    )
    pid = modifier.get_random_puzzle_id()
    p = modifier.get_puzzle(pid)
    p_hard, ok = modifier.make_harder(p, method="add_walls", count=1)
    if ok:
        new_id = modifier.store_modified_puzzle(p_hard, original_id=pid, is_test=False)
# Files written to:
#   ./data/jsonl_out/
#     meta.json
#     train.jsonl   (if is_test=False)
#     test.jsonl    (if is_test=True)
"""

import os
import json
import random
import logging
import numpy as np
import copy
import hashlib
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Iterator, Callable

from .sokoban import (
    Sokoban,
    SokobanTrace,
    generate_level,
    SokobanTraceDataset,
    CellState,
    AStarSokobanState,
    TraceStep,
    TokenizedTrace,  # (kept for compatibility if present in sokoban)
    WithBoxSokobanTokenizer,
    SOKOBAN_DB_NAME,
    astar,
    AStarCannotSolveTaskException,
)
from .trace import TokenizedDataset, TokenizedTrace as TT2, Tokenizer
from .utils import mongodb_client


# -----------------------------
# Lightweight JSONL file store
# -----------------------------

class _JsonlStore:
    """
    Minimal JSONL writer used when `out_dir` is provided to the modifier.

    Layout:
      out_dir/
        meta.json
        train.jsonl
        val.jsonl    # (not used by store_modified_puzzle, but reserved)
        test.jsonl
    """
    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "train": self.root / "train.jsonl",
            "val":   self.root / "val.jsonl",
            "test":  self.root / "test.jsonl",
        }
        meta = self.root / "meta.json"
        if not meta.exists():
            meta.write_text(json.dumps(
                {"format": "sokoban-jsonl-v1", "created_at": time.time()},
                indent=2
            ))

    def add(self, split: str, obj: dict):
        path = self.paths[split]
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")


class SokobanDifficultyModifier:
    """
    A class to modify the difficulty of Sokoban puzzles from existing datasets.

    Methods:
      1. Extract puzzles directly from SokobanTraceDataset (Mongo-based source)
      2. Modify their difficulty (make easier / harder)
      3. Measure difficulty
      4. Store modified puzzles back into:
         - MongoDB (default, legacy behavior), or
         - JSONL files (when `out_dir` is provided)
    """

    def __init__(self, dataset_name: str, create_modified_dataset: bool = True, out_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the difficulty modifier with a dataset.

        Args:
            dataset_name: Name of the existing trace dataset (not tokenized)
            create_modified_dataset: If True, create a new dataset for modified puzzles
            out_dir: If provided, enable JSONL writing to this directory. If None,
                     use MongoDB as before.
        """
        self.dataset_name = dataset_name

        # Extract size parameters from the dataset name
        # Expected format: sokoban.{width}-by-{height}-walls-{num_walls}-boxes-{num_boxes}
        self.width = 7
        self.height = 7
        self.num_walls = 3
        self.num_boxes = 2

        if dataset_name.startswith('sokoban.'):
            size_match = re.search(r'sokoban\.(\d+)-by-(\d+)', dataset_name)
            if size_match:
                self.width = int(size_match.group(1))
                self.height = int(size_match.group(2))
            walls_match = re.search(r'walls-(\d+)', dataset_name)
            if walls_match:
                self.num_walls = int(walls_match.group(1))
            boxes_match = re.search(r'boxes-(\d+)', dataset_name)
            if boxes_match:
                self.num_boxes = int(boxes_match.group(1))

        # Source dataset (Mongo)
        self.trace_dataset = SokobanTraceDataset(dataset_name)

        # Tokenized dataset (Mongo), if available
        tokenized_name = f"{dataset_name}.with-box-40k"
        try:
            self.tok_dataset = TokenizedDataset(tokenized_name)
            self.has_tokenized_dataset = True
        except Exception:
            self.has_tokenized_dataset = False

        # Modified datasets (Mongo), for backward compatibility
        if create_modified_dataset:
            self.modified_dataset_name = f"{dataset_name}.modified"
            self.modified_trace_dataset = SokobanTraceDataset(self.modified_dataset_name)

            if self.has_tokenized_dataset:
                self.modified_tok_dataset_name = f"{self.modified_dataset_name}.with-box-40k"
                try:
                    self.modified_tok_dataset = TokenizedDataset(self.modified_tok_dataset_name)
                except Exception:
                    self.modified_tok_dataset = TokenizedDataset(self.modified_tok_dataset_name)
                    if self.has_tokenized_dataset:
                        self.modified_tok_dataset.add_vocabulary(self.tok_dataset.vocabulary)

        # Tokenizer if tokenized datasets exist
        if self.has_tokenized_dataset:
            self.tokenizer = WithBoxSokobanTokenizer(self.width, self.height)
        else:
            self.tokenizer = None

        # Optional JSONL backend
        self.out_dir: Optional[Path] = Path(out_dir) if out_dir else None
        self._file_store: Optional[_JsonlStore] = _JsonlStore(self.out_dir) if self.out_dir else None

        logging.info(
            f"Initialized SokobanDifficultyModifier for dataset: {dataset_name} "
            f"({self.width}x{self.height}, walls: {self.num_walls}, boxes: {self.num_boxes})"
        )
        if self._file_store:
            logging.info(f"JSONL backend enabled at: {self.out_dir}")

    # ----------------------------
    # Convenience getters (Mongo)
    # ----------------------------
    def get_random_puzzle_id(self, from_test: bool = False) -> int:
        """
        Get a random puzzle ID from the dataset.

        Args:
            from_test: If True, get from test set, otherwise from train set

        Returns:
            Puzzle ID
        """
        client = mongodb_client()
        db = client[SOKOBAN_DB_NAME]
        index_collection = db[f"{self.dataset_name}.index"]

        filtered_docs = index_collection.find({"is_test": from_test}, {"_id": 1})
        filtered_ids = [doc["_id"] for doc in filtered_docs]

        if filtered_ids:
            return random.choice(filtered_ids)

        index_list = self.trace_dataset.index_list
        return random.choice(index_list) if index_list else -1

    def get_puzzle(self, puzzle_id: Union[int, str], from_test: Optional[bool] = None) -> Sokoban:
        """
        Get a puzzle from the dataset by ID.

        Args:
            puzzle_id: The ID of the puzzle to retrieve (int or str)
            from_test: If provided, ensures the puzzle comes from test/train set.
        """
        client = mongodb_client()
        db = client[SOKOBAN_DB_NAME]
        trace_collection = db[f"{self.dataset_name}.trace"]

        try:
            if isinstance(puzzle_id, str) and puzzle_id.isdigit():
                puzzle_id = int(puzzle_id)
        except (ValueError, TypeError):
            pass

        query = {"_id": puzzle_id}
        if from_test is not None:
            query["is_test"] = from_test

        trace_doc = trace_collection.find_one(query)
        if not trace_doc:
            raise ValueError(f"Could not find puzzle with ID {puzzle_id}")

        trace = SokobanTrace.from_dict(trace_doc["trace"])
        return trace.sokoban_start

    # ----------------------------
    # Difficulty operations
    # ----------------------------
    def make_easier(
        self,
        sokoban: Sokoban,
        method: str = "partial_solve",
        percentage: float = 0.3,
        walls_to_remove: int = 1,
    ) -> Tuple[Sokoban, bool]:
        if method == "partial_solve":
            return self._partially_solve_puzzle(sokoban, percentage)
        elif method == "remove_walls":
            return self._remove_walls(sokoban, walls_to_remove)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _partially_solve_puzzle(
        self,
        sokoban: Sokoban,
        solution_percentage: float = 0.3,
    ) -> Tuple[Sokoban, bool]:
        """Create a partially solved version of the puzzle."""
        s_copy = Sokoban(copy.deepcopy(sokoban.state))
        try:
            trace = astar(AStarSokobanState(s_copy, deterministic=False))
            plan_steps = [step for step in trace if step.action == "plan"]
            if len(plan_steps) <= 2:
                return sokoban, False
            num_steps = max(1, int(solution_percentage * (len(plan_steps) - 1)))
            partially_solved = Sokoban(plan_steps[num_steps].state["state"])
            return partially_solved, True
        except AStarCannotSolveTaskException:
            return sokoban, False

    def _remove_walls(
        self,
        sokoban: Sokoban,
        num_walls_to_remove: int = 1,
    ) -> Tuple[Sokoban, bool]:
        """Make a puzzle easier by removing walls."""
        modified = Sokoban(copy.deepcopy(sokoban.state))

        wall_positions: List[Tuple[int, int]] = []
        for y in range(1, sokoban.height - 1):
            for x in range(1, sokoban.width - 1):
                if sokoban.get_state(x, y) == CellState.wall:
                    wall_positions.append((x, y))

        if len(wall_positions) < num_walls_to_remove:
            return sokoban, False

        random.shuffle(wall_positions)
        for i in range(min(num_walls_to_remove, len(wall_positions))):
            x, y = wall_positions[i]
            modified.set_state(x, y, CellState.floor)

        try:
            astar(AStarSokobanState(modified, deterministic=False))
            return modified, True
        except AStarCannotSolveTaskException:
            return sokoban, False

    def make_harder(
        self,
        sokoban: Sokoban,
        method: str = "add_walls",
        count: int = 1,
    ) -> Tuple[Sokoban, bool]:
        if method == "add_walls":
            return self._add_walls(sokoban, count)
        elif method == "add_box":
            return self._add_box(sokoban) if count > 0 else (sokoban, False)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _add_walls(
        self,
        sokoban: Sokoban,
        num_walls_to_add: int = 1,
        avoid_solution_path: bool = True,
    ) -> Tuple[Sokoban, bool]:
        """Make a puzzle harder by adding walls."""
        modified = Sokoban(copy.deepcopy(sokoban.state))

        solution_positions: Set[Tuple[int, int]] = set()
        if avoid_solution_path:
            try:
                trace = astar(AStarSokobanState(sokoban, deterministic=False))
                plan_steps = [step for step in trace if step.action == "plan"]
                for step in plan_steps:
                    wx, wy = Sokoban(step.state["state"]).find_worker()
                    solution_positions.add((wx, wy))
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        solution_positions.add((wx + dx, wy + dy))
            except AStarCannotSolveTaskException:
                avoid_solution_path = False

        possible_positions: List[Tuple[int, int]] = []
        for y in range(1, sokoban.height - 1):
            for x in range(1, sokoban.width - 1):
                if sokoban.get_state(x, y) == CellState.floor:
                    if avoid_solution_path and (x, y) in solution_positions:
                        continue
                    possible_positions.append((x, y))

        if len(possible_positions) < num_walls_to_add:
            return sokoban, False

        random.shuffle(possible_positions)
        for i in range(num_walls_to_add):
            x, y = possible_positions[i]
            modified.set_state(x, y, CellState.wall)

        try:
            astar(AStarSokobanState(modified, deterministic=False))
            return modified, True
        except AStarCannotSolveTaskException:
            return sokoban, False

    def _add_box(self, sokoban: Sokoban) -> Tuple[Sokoban, bool]:
        """Make a puzzle harder by adding a box and dock."""
        modified = Sokoban(copy.deepcopy(sokoban.state))

        box_positions: List[Tuple[int, int]] = []
        for y in range(2, sokoban.height - 2):
            for x in range(2, sokoban.width - 2):
                if sokoban.get_state(x, y) == CellState.floor and sokoban.is_box_movable(x, y):
                    box_positions.append((x, y))

        dock_positions: List[Tuple[int, int]] = []
        for y in range(1, sokoban.height - 1):
            for x in range(1, sokoban.width - 1):
                if sokoban.get_state(x, y) == CellState.floor:
                    dock_positions.append((x, y))

        if not box_positions or not dock_positions:
            return sokoban, False

        random.shuffle(box_positions)
        random.shuffle(dock_positions)
        bx, by = box_positions[0]
        dx, dy = dock_positions[0]

        modified.set_state(bx, by, CellState.box)
        modified.set_state(dx, dy, CellState.dock)

        try:
            astar(AStarSokobanState(modified, deterministic=False))
            return modified, True
        except AStarCannotSolveTaskException:
            return sokoban, False

    # ----------------------------
    # Measuring difficulty
    # ----------------------------
    def measure_difficulty(self, sokoban: Sokoban) -> Dict[str, Any]:
        """
        Measure the difficulty of a Sokoban puzzle.

        Returns a dict with:
          - solution_length
          - exploration_steps
          - solve_time_seconds
          - difficulty_score (higher = harder)
          - is_solvable
          - difficulty_level (easy/medium/hard/impossible)
        """
        try:
            t0 = time.time()
            trace = astar(AStarSokobanState(sokoban, deterministic=False))
            solve_time = time.time() - t0

            plan_steps = [step for step in trace if step.action == "plan"]
            reasoning_steps = [step for step in trace if step.action != "plan"]

            solution_length = len(plan_steps)
            exploration_steps = len(reasoning_steps)
            # Simple proxy; keep close to original code’s spirit
            avg_branching_factor = (len(reasoning_steps) / max(1, len(plan_steps))) if plan_steps else 0.0

            difficulty_score = solution_length
            exploration_factor = min(1.0, exploration_steps / (solution_length * 10 + 1))
            difficulty_score += exploration_factor * 5

            if difficulty_score < 5:
                difficulty_level = "easy"
            elif difficulty_score < 15:
                difficulty_level = "medium"
            else:
                difficulty_level = "hard"

            return {
                "solution_length": solution_length,
                "exploration_steps": exploration_steps,
                "solve_time_seconds": solve_time,
                "avg_branching_factor": avg_branching_factor,
                "difficulty_score": round(difficulty_score, 2),
                "is_solvable": True,
                "difficulty_level": difficulty_level,
            }
        except AStarCannotSolveTaskException:
            return {
                "solution_length": float('inf'),
                "exploration_steps": float('inf'),
                "solve_time_seconds": float('inf'),
                "avg_branching_factor": 0.0,
                "difficulty_score": float('inf'),
                "is_solvable": False,
                "difficulty_level": "impossible",
            }

    # ----------------------------
    # Store modified puzzles
    # ----------------------------
    def store_modified_puzzle(
        self,
        sokoban: Sokoban,
        original_id: Optional[int] = None,
        is_test: bool = False,
    ) -> int:
        """
        Store a modified puzzle back into the dataset.

        If `out_dir` was provided at construction, write to JSONL files.
        Otherwise, store to MongoDB (original behavior).
        """
        try:
            trace_iter = astar(AStarSokobanState(sokoban, deterministic=False))
            sokoban_trace = SokobanTrace(sokoban, list(trace_iter))

            # Generate a unique ID (kept close to original approach)
            if original_id is not None:
                hash_input = f"{original_id}_{time.time()}"
                new_id = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % (10 ** 10)
            else:
                # use a stable-ish hash mod space
                new_id = abs(hash(sokoban_trace)) % (10 ** 10)

            # ---------- JSONL path (preferred when enabled) ----------
            if self._file_store is not None:
                split = "test" if is_test else "train"
                metrics = self.measure_difficulty(sokoban)
                obj = {
                    "_id": new_id,
                    "parent_id": original_id,
                    "split": split,
                    "domain": "in",  # adjust to "out" yourself if writing OOD
                    "spec": {
                        "width": sokoban.width,
                        "height": sokoban.height,
                        "num_walls": None,
                        "num_boxes": None,
                    },
                    "metrics": metrics,
                    "trace": sokoban_trace.to_dict(),
                }
                self._file_store.add(split, obj)
                logging.info(f"Wrote JSONL to {self.out_dir}/{split}.jsonl (_id={new_id})")
                return new_id

            # ---------- Mongo fallback (original behavior) ----------
            client = mongodb_client()
            db = client[SOKOBAN_DB_NAME]

            trace_collection_name = f"{self.modified_dataset_name}.trace"
            index_collection_name = f"{self.modified_dataset_name}.index"

            trace_collection = db[trace_collection_name]
            index_collection = db[index_collection_name]

            trace_doc = {
                "_id": new_id,
                "trace": sokoban_trace.to_dict(),
                "is_test": is_test,
            }
            trace_collection.insert_one(trace_doc)

            index_doc = {
                "_id": new_id,
                "is_test": is_test,
                "creation_time": time.time(),
            }
            index_collection.insert_one(index_doc)

            # Optional: tokenized path (if tokenized datasets exist)
            if self.has_tokenized_dataset and self.tokenizer is not None:
                tokenized_trace: TT2 = self.tokenizer.tokenize(sokoban_trace, is_test=is_test)  # type: ignore
                tokenized_trace = TT2(
                    id=new_id,
                    prompt=tokenized_trace.prompt,
                    reasoning=tokenized_trace.reasoning,
                    plan=tokenized_trace.plan,
                )
                tok_collection_name = f"{self.modified_tok_dataset_name}.{'test' if is_test else 'train'}"
                tok_collection = db[tok_collection_name]
                tok_doc = {
                    "_id": new_id,
                    "prompt": tokenized_trace.prompt,
                    "reasoning": tokenized_trace.reasoning,
                    "plan": tokenized_trace.plan,
                }
                tok_collection.insert_one(tok_doc)

            logging.info(f"Stored modified puzzle in Mongo (_id={new_id})")
            return new_id

        except AStarCannotSolveTaskException:
            logging.error("Cannot store modified puzzle — not solvable")
            return -1

    # ----------------------------
    # Stats helpers (Mongo source)
    # ----------------------------
    def get_puzzle_stats(
        self,
        num_samples: int = 100,
        from_test: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute simple stats from a Mongo-backed source dataset (unchanged).
        """
        client = mongodb_client()
        db = client[SOKOBAN_DB_NAME]
        index_collection = db[f"{self.dataset_name}.index"]

        filtered_docs = index_collection.find({"is_test": from_test}, {"_id": 1})
        filtered_ids = [doc["_id"] for doc in filtered_docs]

        if len(filtered_ids) > num_samples:
            sample_ids = random.sample(filtered_ids, num_samples)
        else:
            sample_ids = filtered_ids

        solution_lengths: List[int] = []
        exploration_steps: List[int] = []
        solve_times: List[float] = []
        difficulty_scores: List[float] = []
        difficulty_levels = {"easy": 0, "medium": 0, "hard": 0, "impossible": 0}

        for puzzle_id in sample_ids:
            try:
                puzzle = self.get_puzzle(puzzle_id, from_test=from_test)
                metrics = self.measure_difficulty(puzzle)

                if metrics["is_solvable"]:
                    solution_lengths.append(metrics["solution_length"])
                    exploration_steps.append(metrics["exploration_steps"])
                    solve_times.append(metrics["solve_time_seconds"])
                    difficulty_scores.append(metrics["difficulty_score"])

                difficulty_levels[metrics["difficulty_level"]] += 1
            except Exception as e:
                logging.error(f"Error processing puzzle {puzzle_id}: {e}")

        total = sum(difficulty_levels.values())
        solvable = total - difficulty_levels["impossible"]

        return {
            "avg_solution_length": float(np.mean(solution_lengths)) if solution_lengths else 0.0,
            "median_solution_length": float(np.median(solution_lengths)) if solution_lengths else 0.0,
            "max_solution_length": int(max(solution_lengths)) if solution_lengths else 0,
            "min_solution_length": int(min(solution_lengths)) if solution_lengths else 0,
            "avg_exploration_steps": float(np.mean(exploration_steps)) if exploration_steps else 0.0,
            "avg_difficulty_score": float(np.mean(difficulty_scores)) if difficulty_scores else 0.0,
            "avg_solve_time": float(np.mean(solve_times)) if solve_times else 0.0,
            "difficulty_distribution": difficulty_levels,
            "num_samples": len(sample_ids),
            "solvable_percentage": 100.0 * solvable / max(1, total),
        }

    def clear_modified_dataset(self):
        """Clear the modified dataset in Mongo (unchanged)."""
        self.modified_trace_dataset.drop()
        if self.has_tokenized_dataset:
            self.modified_tok_dataset.drop()
            self.modified_tok_dataset = TokenizedDataset(self.modified_tok_dataset_name)
            self.modified_tok_dataset.add_vocabulary(self.tok_dataset.vocabulary)
        logging.info(f"Cleared modified datasets: {self.modified_dataset_name}")
