"""
Command-line interface for Snake RL.

Provides easy-to-use commands for training, evaluation, and visualization
using the Click library for a modern CLI experience.

Author: Kacper Kowalski
Date: January 2026
"""

import click
from pathlib import Path
import json
import sys
from typing import Optional

try:
    import click
except ImportError:
    print("Click is required for CLI. Install with: pip install click")
    sys.exit(1)

from Snake import SnakeGame
from QLearningAgent import QLearningAgent
from config import GameConfig, TrainingConfig, VisualizationConfig, PathConfig, PresetConfigs


@click.group()
@click.version_option(version="2.0.0", prog_name="snake-rl")
def cli():
    """Snake Reinforcement Learning CLI - Train AI agents to play Snake."""
    pass


@cli.command()
@click.option("--episodes", "-e", default=10000, help="Number of training episodes")
@click.option("--board-size", "-b", default=16, help="Board size (width=height)")
@click.option("--learning-rate", "-lr", default=0.1, help="Learning rate")
@click.option("--gamma", "-g", default=0.9, help="Discount factor")
@click.option("--epsilon", default=0.2, help="Initial exploration rate")
@click.option("--epsilon-decay", default=0.9995, help="Epsilon decay rate")
@click.option("--model-path", "-m", default=None, help="Path to save model")
@click.option(
    "--preset",
    type=click.Choice(["quick", "standard", "intensive"]),
    help="Use a preset configuration",
)
@click.option("--log-dir", default="runs", help="Directory for experiment logs")
@click.option("--use-wandb/--no-wandb", default=False, help="Use Weights & Biases logging")
@click.option("--use-tensorboard/--no-tensorboard", default=True, help="Use TensorBoard logging")
def train(
    episodes: int,
    board_size: int,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    model_path: Optional[str],
    preset: Optional[str],
    log_dir: str,
    use_wandb: bool,
    use_tensorboard: bool,
):
    """Train a Q-Learning agent to play Snake."""
    from experiment_tracking import ExperimentLogger, MetricsAggregator, PerformanceMonitor
    import numpy as np

    click.echo(click.style("Starting Snake Q-Learning Training", fg="green", bold=True))

    # Load preset if specified
    if preset:
        click.echo(f"Loading preset: {preset}")
        if preset == "quick":
            game_config, training_config = PresetConfigs.quick_test()
        elif preset == "standard":
            game_config, training_config = PresetConfigs.standard_training()
        elif preset == "intensive":
            game_config, training_config = PresetConfigs.intensive_training()
    else:
        game_config = GameConfig(board_width=board_size, board_height=board_size)
        training_config = TrainingConfig(
            num_episodes=episodes, learning_rate=learning_rate, gamma=gamma, epsilon=epsilon
        )

    # Initialize experiment tracking
    config_dict = {
        "board_size": game_config.board_width,
        "episodes": episodes,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon": epsilon,
        "epsilon_decay": epsilon_decay,
    }

    logger = ExperimentLogger(
        experiment_name=f"qlearning_b{board_size}_e{episodes}",
        config=config_dict,
        log_dir=log_dir,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
    )

    # Initialize environment and agent
    game = SnakeGame(width=game_config.board_width, height=game_config.board_height)
    agent = QLearningAgent(
        gamma=gamma, epsilon=epsilon, learning_rate=learning_rate, epsilon_decay=epsilon_decay
    )

    # Training loop
    metrics_agg = MetricsAggregator(window_size=100)
    perf_monitor = PerformanceMonitor()

    best_avg_score = float("-inf")

    with click.progressbar(
        range(episodes), label="Training progress", show_eta=True, show_percent=True
    ) as bar:
        for episode in bar:
            state = game.reset()
            episode_reward = 0
            steps = 0

            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = game.step(action)
                agent.update_q_value(state, action, reward, next_state)

                episode_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            agent.decay_epsilon()

            # Log metrics
            metrics_agg.update(score=episode_reward, steps=steps)
            perf_stats = perf_monitor.update(episode, steps, episode_reward)

            if episode % 100 == 0:
                stats = metrics_agg.get_stats("score")
                avg_score = stats["mean"]

                logger.log_metrics(
                    {
                        "episode": episode,
                        "score": episode_reward,
                        "avg_score": avg_score,
                        "max_score": stats["max"],
                        "epsilon": agent.epsilon,
                        "steps": steps,
                        **perf_stats,
                    },
                    step=episode,
                )

                # Save best model
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    save_path = model_path or f"models/best_model_e{episode}.pkl"
                    agent.save_q_table(save_path)
                    click.echo(f"\nNew best avg score: {avg_score:.2f} (saved to {save_path})")

    # Save final model
    final_path = model_path or "models/final_model.pkl"
    agent.save_q_table(final_path)

    logger.finish()
    click.echo(
        click.style(f"\nTraining complete! Model saved to {final_path}", fg="green", bold=True)
    )


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--episodes", "-e", default=100, help="Number of evaluation episodes")
@click.option("--board-size", "-b", default=16, help="Board size")
@click.option("--visualize/--no-visualize", default=False, help="Show visualization")
def evaluate(model_path: str, episodes: int, board_size: int, visualize: bool):
    """Evaluate a trained model."""
    import numpy as np

    click.echo(click.style(f"Evaluating model: {model_path}", fg="blue", bold=True))

    # Load agent
    agent = QLearningAgent()
    agent.load_q_table(model_path)
    agent.epsilon = 0.0  # Greedy evaluation

    # Create environment
    game = SnakeGame(width=board_size, height=board_size)

    scores = []
    steps_list = []

    with click.progressbar(range(episodes), label="Evaluating") as bar:
        for _ in bar:
            state = game.reset()
            episode_reward = 0
            steps = 0

            while True:
                action = agent.select_action(state, training=False)
                state, reward, done, _ = game.step(action)
                episode_reward += reward
                steps += 1

                if done:
                    break

            scores.append(episode_reward)
            steps_list.append(steps)

    # Display results
    click.echo("\n" + "=" * 50)
    click.echo(click.style("Evaluation Results", fg="cyan", bold=True))
    click.echo("=" * 50)
    click.echo(f"Average Score:  {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    click.echo(f"Max Score:      {np.max(scores)}")
    click.echo(f"Min Score:      {np.min(scores)}")
    click.echo(f"Median Score:   {np.median(scores):.2f}")
    click.echo(f"Average Steps:  {np.mean(steps_list):.2f}")
    click.echo("=" * 50)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--board-size", "-b", default=16, help="Board size")
@click.option("--delay", "-d", default=100, help="Delay between moves (ms)")
def demo(model_path: str, board_size: int, delay: int):
    """Watch a trained agent play Snake interactively."""
    import time

    click.echo(click.style("Starting Snake AI Demo", fg="magenta", bold=True))
    click.echo("Press Ctrl+C to stop\n")

    agent = QLearningAgent()
    agent.load_q_table(model_path)
    agent.epsilon = 0.0

    game = SnakeGame(width=board_size, height=board_size)

    try:
        while True:
            state = game.reset()
            score = 0
            steps = 0

            while True:
                # Render
                board = game.render()
                click.clear()
                click.echo(f"Score: {score} | Steps: {steps}")
                click.echo("-" * (board_size * 2 + 1))
                for row in board:
                    line = (
                        "|"
                        + "".join(
                            "██" if cell == 2 else "**" if cell == 1 else "  " for cell in row
                        )
                        + "|"
                    )
                    click.echo(line)
                click.echo("-" * (board_size * 2 + 1))

                # Step
                action = agent.select_action(state, training=False)
                state, reward, done, _ = game.step(action)
                score += reward
                steps += 1

                time.sleep(delay / 1000.0)

                if done:
                    click.echo(
                        click.style(f"\nGame Over! Final Score: {score}", fg="red", bold=True)
                    )
                    time.sleep(2)
                    break

    except KeyboardInterrupt:
        click.echo("\n\nDemo stopped.")


@cli.command()
@click.option("--output", "-o", default="config.json", help="Output file path")
@click.option("--preset", type=click.Choice(["quick", "standard", "intensive"]))
def export_config(output: str, preset: Optional[str]):
    """Export configuration to JSON file."""
    if preset:
        if preset == "quick":
            game_config, training_config = PresetConfigs.quick_test()
        elif preset == "standard":
            game_config, training_config = PresetConfigs.standard_training()
        else:
            game_config, training_config = PresetConfigs.intensive_training()
    else:
        game_config = GameConfig()
        training_config = TrainingConfig()

    config = {"game": game_config.__dict__, "training": training_config.__dict__}

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(config, f, indent=2)

    click.echo(f"Configuration exported to {output}")


@cli.command()
def info():
    """Display system and package information."""
    import numpy as np
    import matplotlib

    click.echo(click.style("\nSnake RL System Information", fg="cyan", bold=True))
    click.echo("=" * 50)
    click.echo(f"Python Version:     {sys.version.split()[0]}")
    click.echo(f"NumPy Version:      {np.__version__}")
    click.echo(f"Matplotlib Version: {matplotlib.__version__}")

    try:
        import torch

        click.echo(f"PyTorch Version:    {torch.__version__}")
        click.echo(f"CUDA Available:     {torch.cuda.is_available()}")
    except ImportError:
        click.echo("PyTorch:            Not installed")

    try:
        import wandb

        click.echo(f"W&B Version:        {wandb.__version__}")
    except ImportError:
        click.echo("W&B:                Not installed")

    click.echo("=" * 50)


if __name__ == "__main__":
    cli()
