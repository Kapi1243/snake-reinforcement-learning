"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def seed_random():
    """Fixture to seed random number generators for reproducibility."""
    np.random.seed(42)
    import random

    random.seed(42)
    yield
    # Reset after test
    np.random.seed(None)


@pytest.fixture
def small_game():
    """Fixture providing a small game instance for testing."""
    from src.Snake import SnakeGame

    return SnakeGame(width=8, height=8)


@pytest.fixture
def default_agent():
    """Fixture providing a default Q-learning agent."""
    from src.QLearningAgent import QLearningAgent

    return QLearningAgent()


@pytest.fixture
def trained_agent(default_agent, small_game):
    """Fixture providing a partially trained agent."""
    agent = default_agent
    game = small_game

    # Train for a few episodes
    for _ in range(100):
        state = game.reset()
        for _ in range(50):
            action = agent.select_action(state)
            next_state, reward, done, _ = game.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            if done:
                break

    return agent
