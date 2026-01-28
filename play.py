"""
Train an AI to play Snake and see the results.
Just run: python play.py
"""

import numpy as np
from pathlib import Path
from src.Snake import SnakeGame
from src.QLearningAgent import QLearningAgent

def main():
    print("=" * 60)
    print("SNAKE REINFORCEMENT LEARNING - MULTI-SEED EVALUATION")
    print("=" * 60)
    
    # Configuration
    experiment_name = "Test 5 - Higher Epsilon"
    config = {
        'gamma': 0.9,
        'learning_rate': 0.3,
        'epsilon': 0.4,
        'board_size': 12,
        'num_episodes': 5000
    }
    
    # Seeds for reproducibility
    seeds = [42, 123, 456, 789, 2024]
    num_eval_games = 100
    
    all_avg_scores = []
    all_best_scores = []
    all_fail_rates = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {i+1}/5: {seed}")
        print(f"{'='*60}")
        
        np.random.seed(seed)
        
        # Train the agent
        agent = QLearningAgent(
            gamma=config['gamma'],
            learning_rate=config['learning_rate'],
            epsilon=config['epsilon']
        )
        
        agent.train(
            board_size=config['board_size'],
            num_episodes=config['num_episodes'],
            eval_frequency=100,
            eval_episodes=50,
            save_q_history=False,
            verbose=False
        )
        
        # Evaluate
        print(f"\nEvaluating with seed {seed}...")
        avg_score, scores = agent.evaluate(board_size=config['board_size'], num_episodes=num_eval_games)
        best_score = max(scores)
        worst_score = min(scores)
        fail_rate = sum(1 for s in scores if s < 5) / num_eval_games * 100
        
        all_avg_scores.append(avg_score)
        all_best_scores.append(best_score)
        all_fail_rates.append(fail_rate)
        
        print(f"  Avg: {avg_score:.2f} | Best: {best_score} | Worst: {worst_score} | Fail rate: {fail_rate:.1f}%")
    
    # Aggregate statistics
    print("\n" + "=" * 60)
    print("FINAL RESULTS ACROSS ALL SEEDS")
    print("=" * 60)
    
    avg_mean = np.mean(all_avg_scores)
    avg_std = np.std(all_avg_scores)
    median_best = np.median(all_best_scores)
    mean_fail_rate = np.mean(all_fail_rates)
    
    print(f"\nAvg score over eval: {avg_mean:.2f} ± {avg_std:.2f}")
    print(f"Best score (median): {median_best:.0f}")
    print(f"Fail rate (<5): {mean_fail_rate:.1f}%")
    
    print(f"\nDetailed breakdown:")
    print(f"  Individual avg scores: {[f'{s:.2f}' for s in all_avg_scores]}")
    print(f"  Individual best scores: {all_best_scores}")
    print(f"  Individual fail rates: {[f'{r:.1f}%' for r in all_fail_rates]}")

if __name__ == "__main__":
    main()
