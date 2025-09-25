#!/usr/bin/env python3
"""
Quick test script for Sokoban integration in tiny-grpo.
This tests that we can load levels and evaluate completions.
"""

from train import load_sokoban_levels, make_sokoban_reward_fn, parse_moves_from_answer

def test_sokoban_integration():
    print("Testing Sokoban integration...")
    
    # Load levels
    levels = load_sokoban_levels(max_rows=1)
    print(f"✓ Loaded {len(levels)} Sokoban levels")
    
    if not levels:
        print("❌ No levels found! Check that searchformer-main/static/sokoban/*.txt exists")
        return
    
    # Test first level
    level = levels[0]
    print(f"✓ Level file: {level['level_file']}")
    print(f"✓ Task prompt (first 100 chars): {level['task'][:100]}...")
    
    # Test reward function
    reward_fn = make_sokoban_reward_fn(level['level'])
    
    # Test various completions
    test_cases = [
        ("<answer>right right up left down</answer>", "Valid format with moves"),
        ("<answer>invalid moves here</answer>", "Invalid moves"),
        ("no answer tags", "No answer tags"),
        ("<answer>up down left right</answer>", "Different valid moves"),
        ("<think>reasoning</think> <answer>left right</answer>", "With thinking"),
    ]
    
    print("\nTesting reward function:")
    for completion, description in test_cases:
        reward = reward_fn(completion)
        moves = parse_moves_from_answer(completion.replace("<answer>", "").replace("</answer>", ""))
        print(f"  {description}: reward={reward:.1f}, moves={moves}")
    
    print("\n✓ All tests completed successfully!")
    print("Ready to run: python train.py")

if __name__ == "__main__":
    test_sokoban_integration()