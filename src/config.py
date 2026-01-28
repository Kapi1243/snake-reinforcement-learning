"""
Configuration file for Snake Q-Learning project.

This module centralizes all hyperparameters and settings for easy experimentation
and reproducibility.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class GameConfig:
    """Configuration for the Snake game environment."""

    board_width: int = 16
    board_height: int = 16

    # Reward values
    reward_food: float = 10.0
    reward_toward_food: float = 1.0
    reward_collision: float = -10.0

    # Game mechanics
    max_steps_without_food: int = 100  # Prevent infinite loops


@dataclass
class TrainingConfig:
    """Configuration for Q-Learning training."""

    # Q-Learning hyperparameters
    gamma: float = 0.8  # Discount factor
    epsilon: float = 0.2  # Exploration rate
    learning_rate: float = 0.9  # Learning rate (for incremental updates)

    # Training settings
    num_episodes: int = 10000  # Total training episodes
    eval_frequency: int = 100  # Evaluate every N episodes
    eval_episodes: int = 25  # Number of episodes per evaluation

    # State space
    num_states: int = 256  # 2^8 possible states
    num_actions: int = 4  # 4 movement directions

    # Logging and saving
    save_frequency: int = 1000  # Save model every N episodes
    save_q_history: bool = False  # Save Q-table history (for convergence plots)
    verbose: bool = True  # Print training progress


@dataclass
class VisualizationConfig:
    """Configuration for visualizations and animations."""

    # Animation settings
    animation_interval: int = 75  # Milliseconds between frames
    animation_fps: int = 15  # Frames per second for saved videos
    max_animation_frames: int = 1000

    # Plot settings
    figure_dpi: int = 300
    plot_style: str = "seaborn-v0_8-darkgrid"

    # Episodes to showcase in training animation
    showcase_episodes: Tuple[int, ...] = (0, 200, 400, 600, 800, 1000, 2500, 5000, 10000)


@dataclass
class PathConfig:
    """Configuration for file paths."""

    models_dir: str = "models"
    results_dir: str = "results"
    images_dir: str = "Images"

    default_model_name: str = "q_learning_snake.pkl"
    training_plot_name: str = "training_progress.png"
    convergence_plot_name: str = "ConvergenceGraph.png"
    animation_name: str = "AnimatedGames.gif"


# Create default configurations
DEFAULT_GAME_CONFIG = GameConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_VIZ_CONFIG = VisualizationConfig()
DEFAULT_PATH_CONFIG = PathConfig()


# Preset configurations for different scenarios
class PresetConfigs:
    """Pre-defined configuration presets for common use cases."""

    @staticmethod
    def quick_test() -> Tuple[GameConfig, TrainingConfig]:
        """Fast training for testing purposes."""
        game_config = GameConfig(board_width=8, board_height=8)
        training_config = TrainingConfig(num_episodes=1000, eval_frequency=100, verbose=True)
        return game_config, training_config

    @staticmethod
    def standard_training() -> Tuple[GameConfig, TrainingConfig]:
        """Standard training configuration."""
        return DEFAULT_GAME_CONFIG, DEFAULT_TRAINING_CONFIG

    @staticmethod
    def intensive_training() -> Tuple[GameConfig, TrainingConfig]:
        """Long training for best results."""
        training_config = TrainingConfig(
            num_episodes=50000, eval_frequency=500, save_q_history=True
        )
        return DEFAULT_GAME_CONFIG, training_config

    @staticmethod
    def large_board() -> Tuple[GameConfig, TrainingConfig]:
        """Training on larger board (more challenging)."""
        game_config = GameConfig(board_width=24, board_height=24)
        training_config = TrainingConfig(
            num_episodes=20000, epsilon=0.3, eval_frequency=200  # More exploration needed
        )
        return game_config, training_config
