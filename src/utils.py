"""
Utility functions for visualization and analysis.

This module provides helper functions for creating visualizations,
analyzing training progress, and generating reports.
"""

from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pickle
from pathlib import Path


def plot_training_curves(
    training_history: List[dict], save_path: Optional[str] = None, show: bool = True
) -> Figure:
    """
    Create comprehensive training visualization with multiple subplots.

    Args:
        training_history: List of training statistics per episode
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    if not training_history:
        raise ValueError("Training history is empty")

    episodes = [h["episode"] for h in training_history]
    avg_scores = [h["avg_score"] for h in training_history]
    min_scores = [h["min_score"] for h in training_history]
    max_scores = [h["max_score"] for h in training_history]
    std_scores = [h["std_score"] for h in training_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Q-Learning Training Analysis", fontsize=16, fontweight="bold")

    # 1. Average score over time
    ax1 = axes[0, 0]
    ax1.plot(episodes, avg_scores, "b-", linewidth=2, label="Average Score")
    ax1.fill_between(episodes, min_scores, max_scores, alpha=0.2, label="Min-Max Range")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score (Snake Length)")
    ax1.set_title("Learning Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Score variance over time
    ax2 = axes[0, 1]
    ax2.plot(episodes, std_scores, "r-", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Standard Deviation")
    ax2.set_title("Score Stability")
    ax2.grid(True, alpha=0.3)

    # 3. Moving average
    ax3 = axes[1, 0]
    window = min(10, len(avg_scores) // 10)
    if window > 1:
        moving_avg = np.convolve(avg_scores, np.ones(window) / window, mode="valid")
        moving_episodes = episodes[: len(moving_avg)]
        ax3.plot(episodes, avg_scores, "b-", alpha=0.3, label="Raw")
        ax3.plot(moving_episodes, moving_avg, "b-", linewidth=2, label=f"{window}-Episode MA")
    else:
        ax3.plot(episodes, avg_scores, "b-", linewidth=2)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Average Score")
    ax3.set_title("Smoothed Learning Curve")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Score distribution histogram (final 20% of training)
    ax4 = axes[1, 1]
    final_chunk = int(len(avg_scores) * 0.8)
    final_scores = avg_scores[final_chunk:]
    ax4.hist(final_scores, bins=20, edgecolor="black", alpha=0.7)
    ax4.axvline(
        np.mean(final_scores),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(final_scores):.1f}",
    )
    ax4.set_xlabel("Average Score")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Final Performance Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_q_value_convergence(
    q_history: List[np.ndarray],
    save_path: Optional[str] = None,
    max_lines: int = 50,
    show: bool = True,
) -> Figure:
    """
    Plot convergence of Q-values over training episodes.

    Shows how individual Q(s,a) values change and stabilize during training.

    Args:
        q_history: List of Q-tables at different episodes
        save_path: Optional path to save the figure
        max_lines: Maximum number of Q-value trajectories to plot
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    if not q_history:
        raise ValueError("Q-value history is empty")

    # Find Q-values that were actually used (non-zero at some point)
    used_q_trajectories = []

    for state in range(q_history[0].shape[0]):
        for action in range(q_history[0].shape[1]):
            trajectory = [q_table[state, action] for q_table in q_history]

            # Only plot if this Q-value was ever non-zero
            if any(val != 0 for val in trajectory) and trajectory[-1] != 0:
                used_q_trajectories.append(np.abs(trajectory))

    # Limit number of trajectories for clarity
    if len(used_q_trajectories) > max_lines:
        indices = np.linspace(0, len(used_q_trajectories) - 1, max_lines, dtype=int)
        used_q_trajectories = [used_q_trajectories[i] for i in indices]

    # Normalize each trajectory by its final value
    fig, ax = plt.subplots(figsize=(10, 6))

    for trajectory in used_q_trajectories:
        final_val = np.mean(trajectory[-20:])  # Average of last 20 values
        if final_val > 0:
            normalized = (trajectory - final_val) / final_val
            ax.plot(normalized, color="black", alpha=0.05, linewidth=0.8)

    ax.set_ylim(-1, 0.5)
    ax.set_xlabel("Episode Number", fontsize=12)
    ax.set_ylabel("Normalized Q-Value\n(Relative to Converged Value)", fontsize=12)
    ax.set_title("Q-Table Convergence", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5, linewidth=1.5, label="Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Convergence plot saved to {save_path}")

    if show:
        plt.show()

    return fig


def compare_agents(
    agent_paths: List[Tuple[str, str]],
    board_size: int = 16,
    num_episodes: int = 100,
    save_path: Optional[str] = None,
) -> None:
    """
    Compare performance of multiple trained agents.

    Args:
        agent_paths: List of (name, filepath) tuples for each agent
        board_size: Game board size for evaluation
        num_episodes: Number of episodes to evaluate each agent
        save_path: Optional path to save comparison plot
    """
    from QLearningAgent import QLearningAgent

    results = {}

    for name, path in agent_paths:
        agent = QLearningAgent()
        agent.load(path)
        avg_score, scores = agent.evaluate(board_size, num_episodes)
        results[name] = {
            "avg": avg_score,
            "scores": scores,
            "min": min(scores),
            "max": max(scores),
            "std": np.std(scores),
        }

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of average scores
    names = list(results.keys())
    avgs = [results[n]["avg"] for n in names]
    stds = [results[n]["std"] for n in names]

    ax1.bar(names, avgs, yerr=stds, capsize=5, alpha=0.7, edgecolor="black")
    ax1.set_ylabel("Average Score")
    ax1.set_title("Agent Performance Comparison")
    ax1.grid(True, alpha=0.3, axis="y")

    # Box plot of score distributions
    score_lists = [results[n]["scores"] for n in names]
    ax2.boxplot(score_lists, labels=names)
    ax2.set_ylabel("Score Distribution")
    ax2.set_title("Score Variability")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison saved to {save_path}")

    plt.show()

    # Print statistics
    print("\n" + "=" * 60)
    print("Agent Comparison Results")
    print("=" * 60)
    for name in names:
        print(f"\n{name}:")
        print(f"  Average: {results[name]['avg']:.2f}")
        print(f"  Std Dev: {results[name]['std']:.2f}")
        print(f"  Range: [{results[name]['min']}, {results[name]['max']}]")


def analyze_state_coverage(q_table: np.ndarray) -> dict:
    """
    Analyze which states were visited during training.

    Args:
        q_table: Trained Q-table

    Returns:
        Dictionary with coverage statistics
    """
    total_states = q_table.shape[0]

    # States where at least one action has non-zero Q-value
    visited_states = np.any(q_table != 0, axis=1)
    num_visited = np.sum(visited_states)

    # States where all actions have been tried
    fully_explored = np.all(q_table != 0, axis=1)
    num_fully_explored = np.sum(fully_explored)

    coverage = {
        "total_states": total_states,
        "visited_states": num_visited,
        "fully_explored_states": num_fully_explored,
        "visit_percentage": 100 * num_visited / total_states,
        "full_exploration_percentage": 100 * num_fully_explored / total_states,
    }

    return coverage


def print_training_summary(training_results: dict, coverage: dict) -> None:
    """
    Print a formatted summary of training results.

    Args:
        training_results: Dictionary from agent.train()
        coverage: Dictionary from analyze_state_coverage()
    """
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    print(f"\nPerformance:")
    print(f"  Best Average Score: {training_results['best_avg_score']:.2f}")
    print(f"  Final Average Score: {training_results['final_avg_score']:.2f}")

    print(f"\nState Space Coverage:")
    print(f"  Total Possible States: {coverage['total_states']}")
    print(f"  Visited States: {coverage['visited_states']} ({coverage['visit_percentage']:.1f}%)")
    print(
        f"  Fully Explored: {coverage['fully_explored_states']} ({coverage['full_exploration_percentage']:.1f}%)"
    )

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities loaded.")
    print("Use these functions in your training scripts for better analysis.")
