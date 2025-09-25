from collections.abc import Callable
import json
from pathlib import Path
import os
import sys
import random
import re
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


# --- Sokoban helpers (import from searchformer) ---
# Add searchformer-main to path and import the existing Sokoban class
repo_root = Path(__file__).resolve().parents[1]
searchformer_path = repo_root / "searchformer-main"
if str(searchformer_path) not in sys.path:
    sys.path.insert(0, str(searchformer_path))

from searchformer.sokoban import Sokoban


def sokoban_prompt_from_level(level_str: str) -> str:
    """
    Create a concise Sokoban instruction prompt for chat format.
    The model must output a plan enclosed in <answer>...</answer> using moves: up, down, left, right.
    """
    return (
        "Solve the Sokoban puzzle below. The grid uses these symbols: #=wall, .=dock, $=box, *=box on dock, @=worker, +=worker on dock, space=floor.\n"
        "Return ONLY a sequence of moves as space-separated tokens inside <answer>...</answer>, using the words: up, down, left, right.\n"
        "You may include <think>...</think> before the answer.\n\n"
        f"Level:\n{level_str.strip()}\n\n"
        "Example: <think>reasoning</think> <answer>right right up left down</answer>"
    )


def parse_moves_from_answer(answer_text: str) -> list[str]:
    """Parse moves from free-form text; accepts up/down/left/right or u/d/l/r."""
    toks = re.findall(r"[A-Za-z]+", answer_text)
    mapping = {
        "u": "up",
        "up": "up",
        "d": "down",
        "down": "down",
        "l": "left",
        "left": "left",
        "r": "right",
        "right": "right",
    }
    moves: list[str] = []
    for t in toks:
        t_l = t.lower()
        if t_l in mapping:
            moves.append(mapping[t_l])
    return moves


def make_sokoban_reward_fn(level_str: str) -> Callable[[str], float]:
    """
    Build a reward function that evaluates a generated completion by simulating
    the extracted moves on the given Sokoban level. Returns 1.0 if solved, else 0.0.
    """
    # Prepare initial state once
    grid = [list(line.rstrip("\n")) for line in level_str.strip("\n").splitlines()]

    def reward_fn(completion: str) -> float:
        # Extract <answer>...</answer>
        m = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        if not m:
            return 0.0
        ans = m.group(1)
        moves = parse_moves_from_answer(ans)
        # If no recognizable moves, tiny penalty to encourage output format
        if not moves:
            return 0.0
        sok = Sokoban([row[:] for row in grid])
        for mv in moves:
            try:
                sok.move(mv)
            except Exception:
                # Invalid action token; stop early
                break
        return 1.0 if sok.is_complete else 0.0

    return reward_fn


def load_sokoban_levels_from_json_file(json_file_path: str, max_rows: Optional[int] = None) -> list[dict]:
    """Load Sokoban levels from a specific JSON file."""
    json_path = Path(json_file_path)
    
    if not json_path.exists():
        print(f"Warning: JSON file not found at {json_path}")
        return []
    
    rows: list[dict] = []
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Try to extract Sokoban grids from the JSON data
        if isinstance(data, dict):
            for key, value in data.items():
                # Look for grid-like data structures
                level_str = extract_sokoban_grid(value)
                if level_str:
                    rows.append({
                        "task": sokoban_prompt_from_level(level_str),
                        "level": level_str,
                        "level_file": str(json_path),
                        "key": key
                    })
                    
                    if max_rows is not None and len(rows) >= max_rows:
                        return rows
                        
    except Exception as e:
        print(f"Warning: Could not read {json_path}: {e}")
        return []
    
    return rows


def extract_sokoban_grid(data: Any) -> Optional[str]:
    """Extract Sokoban grid from various data formats."""
    if isinstance(data, str):
        # Check if it looks like a Sokoban grid
        if '#' in data and any(c in data for c in '@$.*+'):
            return data
    elif isinstance(data, dict):
        # Look for common field names that might contain the grid
        for field in ['level', 'grid', 'state', 'sokoban_start', 'puzzle']:
            if field in data:
                sub_data = data[field]
                if isinstance(sub_data, str):
                    return sub_data
                elif isinstance(sub_data, list) and len(sub_data) > 0:
                    # Try to reconstruct grid from list format
                    if all(isinstance(row, str) for row in sub_data):
                        return '\n'.join(sub_data)
                    elif all(isinstance(row, list) for row in sub_data):
                        return '\n'.join(''.join(cell for cell in row) for row in sub_data)
    elif isinstance(data, list) and len(data) > 0:
        # Grid might be stored as list of strings or list of lists
        if all(isinstance(row, str) for row in data):
            return '\n'.join(data)
        elif all(isinstance(row, list) for row in data):
            return '\n'.join(''.join(str(cell) for cell in row) for row in data)
    
    return None


def load_sokoban_levels_from_txt(levels_dir: Optional[str] = None, max_rows: Optional[int] = None) -> list[dict]:
    """Load Sokoban levels from plain text files."""
    if levels_dir is None:
        # default to searchformer-main/static/sokoban
        repo_root = Path(__file__).resolve().parents[1]
        levels_path = repo_root / "searchformer-main" / "static" / "sokoban"
    else:
        levels_path = Path(levels_dir)
    
    if not levels_path.exists():
        print(f"Warning: Sokoban levels directory not found at {levels_path}")
        return []
        
    level_files = sorted(levels_path.glob("*.txt"))
    if not level_files:
        print(f"Warning: No .txt files found in {levels_path}")
        return []
        
    rows: list[dict] = []
    for p in level_files:
        level_str = p.read_text(encoding="utf-8")
        rows.append({
            "task": sokoban_prompt_from_level(level_str),
            "level": level_str,
            "level_file": str(p),
        })
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def load_sokoban_levels(levels_source: Optional[str] = None, max_rows: Optional[int] = None) -> list[dict]:
    """
    Load Sokoban levels from either:
    - A specific JSON file (if levels_source ends with .json)
    - A directory of text files (otherwise)
    """
    if levels_source and levels_source.endswith('.json'):
        # Load from specific JSON file
        return load_sokoban_levels_from_json_file(levels_source, max_rows)
    else:
        # Load from text files directory
        return load_sokoban_levels_from_txt(levels_source, max_rows)


@torch.no_grad()
def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    reward_fn: Optional[Callable[[str], float]] = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("cuda")

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        if reward_fn is not None:
            reward = float(reward_fn(completion))
            returns[i] = reward
        else:
            # search answer tag
            answer_match = re.search(
                r"<answer>(.*?)</answer>",
                completion,
                flags=re.DOTALL,
            )

            answer = answer_match.group(1) if answer_match else None
            reward = 0
            if answer is not None:
                if answer == oracle_answer:
                    reward = 1.0
                elif oracle_answer in answer:
                    reward = 0.5
                else:
                    reward = 0.01

            returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: LlamaForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def main():
    seed = 42
    wandb_project = "tiny_grpo"  # "tiny_grpo"
    device_index = 0
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = 16
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 12
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    reference_model, _ = load_model(model_name, device_map=device)
    model, tokenizer = load_model(model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    # Select dataset mode via env var to keep changes minimal; default to Sokoban
    dataset_mode = os.environ.get("TINY_GRPO_DATASET", "sokoban").lower()

    if dataset_mode == "math":
        prompts = read_prompts(
            "data/math_tasks.jsonl",
            predicate=lambda x: len(x["question"]) < 128
            and x["num_terms"] <= 3
            and x["num_digits"] <= 3,
            max_rows=64 * 1024,
        )
        print(f"found {len(prompts)} matching math prompts")
        prompt_loader = DataLoader(
            prompts,
            batch_size=rollouts_per_step,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
        )
    else:
        # Sokoban mode: load levels. Use SOKOBAN_DATA_FILE for specific JSON file or SOKOBAN_LEVELS_DIR for text files
        sokoban_source = os.environ.get("SOKOBAN_DATA_FILE") or os.environ.get("SOKOBAN_LEVELS_DIR")
        prompts = load_sokoban_levels(levels_source=sokoban_source, max_rows=64 * 1024)
        print(f"found {len(prompts)} sokoban levels")
        prompt_loader = DataLoader(
            prompts,
            batch_size=rollouts_per_step,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
        )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []

        replay_buffer.clear()

        if dataset_mode == "math":
            questions = prompt_batch["question"]
            answers = prompt_batch["answer"]

            with torch.no_grad():
                for q, a in zip(questions, answers):
                    sequence_ids, returns, action_mask, completions = rollout(
                        model,
                        tokenizer,
                        q,
                        a,
                        num_rollouts=group_size,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    print(
                        f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                    )
                    rollout_returns.append(returns.cpu())

                    advantages = group_advantages(returns)
                    attention_mask = sequence_ids != pad_token_id

                    log_probs = sequences_log_probs(
                        model=model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                    )
                    log_probs_ref = sequences_log_probs(
                        model=reference_model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                    )
                    kl = approx_kl_divergence(
                        log_probs=log_probs,
                        log_probs_ref=log_probs_ref,
                        action_mask=action_mask,
                    )

                    experience = Experience(
                        sequences=sequence_ids,
                        action_log_probs=log_probs,
                        log_probs_ref=log_probs_ref,
                        returns=returns,
                        advantages=advantages,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        kl=kl,
                    )
                    replay_buffer.append(experience.to(cpu_device))
        else:
            tasks = prompt_batch["task"]
            levels = prompt_batch["level"]

            with torch.no_grad():
                for task, level_str in zip(tasks, levels):
                    r_fn = make_sokoban_reward_fn(level_str)
                    sequence_ids, returns, action_mask, completions = rollout(
                        model,
                        tokenizer,
                        task,
                        oracle_answer="",
                        num_rollouts=group_size,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        reward_fn=r_fn,
                    )

                    solved = returns.sum().item()
                    print(
                        f"sokoban rollout returns={solved:.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                    )
                    rollout_returns.append(returns.cpu())

                    advantages = group_advantages(returns)
                    attention_mask = sequence_ids != pad_token_id

                    log_probs = sequences_log_probs(
                        model=model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                    )
                    log_probs_ref = sequences_log_probs(
                        model=reference_model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                    )
                    kl = approx_kl_divergence(
                        log_probs=log_probs,
                        log_probs_ref=log_probs_ref,
                        action_mask=action_mask,
                    )

                    experience = Experience(
                        sequences=sequence_ids,
                        action_log_probs=log_probs,
                        log_probs_ref=log_probs_ref,
                        returns=returns,
                        advantages=advantages,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        kl=kl,
                    )
                    replay_buffer.append(experience.to(cpu_device))

        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"returns of step {k}: {episode_return_sum:.4f}")
        wandb.log({"returns": episode_return_sum})

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp: Experience

                exp = exp.to(device)

                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                wandb.log({"kl": kl, "grad_norm": grad_norm})

                optimizer.step()

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    main()
