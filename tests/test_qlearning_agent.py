"""
Unit tests for the Q-Learning agent.

Tests cover Q-value updates, action selection, training mechanics,
model persistence, and learning behavior.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.QLearningAgent import QLearningAgent
from src.Snake import SnakeGame


class TestQLearningAgent:
    """Tests for the Q-Learning agent."""

    def test_initialization(self):
        """Test agent initialization with default parameters."""
        agent = QLearningAgent()

        assert agent.num_states == 256
        assert agent.num_actions == 4
        assert agent.q_table.shape == (256, 4)
        assert np.all(agent.q_table == 0)  # Should initialize to zeros

    def test_custom_hyperparameters(self):
        """Test initialization with custom hyperparameters."""
        agent = QLearningAgent(gamma=0.95, epsilon=0.3, learning_rate=0.2)

        assert agent.gamma == 0.95
        assert agent.epsilon == 0.3
        assert agent.learning_rate == 0.2

    def test_select_action_greedy(self):
        """Test greedy action selection (no exploration)."""
        agent = QLearningAgent(epsilon=0.0)

        # Set Q-values to favor action 2
        agent.q_table[0, :] = [1.0, 2.0, 5.0, 3.0]

        # Should always select action 2 (highest Q-value)
        actions = [agent.select_action(0, training=True) for _ in range(100)]
        assert all(action == 2 for action in actions)

    def test_select_action_exploration(self):
        """Test epsilon-greedy exploration."""
        agent = QLearningAgent(epsilon=1.0)  # Always explore

        # Even with clear best action, should explore randomly
        agent.q_table[0, :] = [1.0, 2.0, 5.0, 3.0]

        actions = [agent.select_action(0, training=True) for _ in range(100)]

        # Should see variety in actions due to exploration
        unique_actions = set(actions)
        assert len(unique_actions) > 1

    def test_select_action_evaluation_mode(self):
        """Test that evaluation mode uses greedy policy."""
        agent = QLearningAgent(epsilon=1.0)

        # Set Q-values
        agent.q_table[0, :] = [1.0, 2.0, 5.0, 3.0]

        # In evaluation mode (training=False), should be greedy
        actions = [agent.select_action(0, training=False) for _ in range(100)]
        assert all(action == 2 for action in actions)

    def test_update_q_value(self):
        """Test Q-value update mechanism."""
        agent = QLearningAgent(learning_rate=1.0, gamma=0.9)

        # Initial Q-value
        assert agent.q_table[0, 0] == 0.0

        # Update: Q(s, a) = r + γ * max(Q(s', a'))
        # With learning_rate=1.0, update is full replacement
        agent.update_q_value(state=0, action=0, reward=10.0, next_state=1)

        # Q(0, 0) should now be: 0 + 1.0 * (10.0 + 0.9 * 0 - 0) = 10.0
        assert agent.q_table[0, 0] == pytest.approx(10.0)

    def test_update_q_value_with_next_state_value(self):
        """Test Q-value update considering next state value."""
        agent = QLearningAgent(learning_rate=1.0, gamma=0.9)

        # Set next state Q-values
        agent.q_table[1, :] = [5.0, 10.0, 3.0, 2.0]

        # Update current state
        agent.update_q_value(state=0, action=0, reward=5.0, next_state=1)

        # Q(0, 0) = 5.0 + 0.9 * max([5.0, 10.0, 3.0, 2.0]) = 5.0 + 0.9 * 10.0 = 14.0
        assert agent.q_table[0, 0] == pytest.approx(14.0)

    def test_epsilon_decay(self):
        """Test epsilon decay mechanism."""
        agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1)

        initial_epsilon = agent.epsilon

        # Decay epsilon multiple times
        for _ in range(100):
            agent.decay_epsilon()

        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.min_epsilon

    def test_epsilon_minimum_bound(self):
        """Test that epsilon doesn't go below minimum."""
        agent = QLearningAgent(epsilon=0.15, epsilon_decay=0.5, min_epsilon=0.1)

        # Decay many times
        for _ in range(1000):
            agent.decay_epsilon()

        assert agent.epsilon == pytest.approx(agent.min_epsilon)


class TestAgentTraining:
    """Tests for agent training functionality."""

    def test_train_single_episode(self):
        """Test training for a single episode."""
        agent = QLearningAgent()
        game = SnakeGame(width=8, height=8)

        # Q-table should be all zeros initially
        assert np.all(agent.q_table == 0)

        # Train one episode
        total_reward = 0
        state = game.reset()

        for _ in range(100):
            action = agent.select_action(state)
            next_state, reward, done, _ = game.step(action)
            agent.update_q_value(state, action, reward, next_state)

            total_reward += reward
            state = next_state

            if done:
                break

        # Q-table should have been updated
        assert not np.all(agent.q_table == 0)

    def test_learning_improves_performance(self):
        """Test that agent performance improves with training."""
        agent = QLearningAgent(epsilon=0.1)
        game = SnakeGame(width=8, height=8)

        # Evaluate before training
        early_scores = []
        for _ in range(10):
            state = game.reset()
            episode_reward = 0
            for _ in range(100):
                action = agent.select_action(state, training=False)
                state, reward, done, _ = game.step(action)
                episode_reward += reward
                if done:
                    break
            early_scores.append(episode_reward)

        early_avg = np.mean(early_scores)

        # Train agent
        for episode in range(500):
            state = game.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, _ = game.step(action)
                agent.update_q_value(state, action, reward, next_state)
                state = next_state
                if done:
                    break
            agent.decay_epsilon()

        # Evaluate after training
        late_scores = []
        for _ in range(10):
            state = game.reset()
            episode_reward = 0
            for _ in range(100):
                action = agent.select_action(state, training=False)
                state, reward, done, _ = game.step(action)
                episode_reward += reward
                if done:
                    break
            late_scores.append(episode_reward)

        late_avg = np.mean(late_scores)

        # Performance should improve (or at least not get significantly worse)
        # Using a lenient threshold since training is stochastic
        assert late_avg >= early_avg * 0.8


class TestModelPersistence:
    """Tests for saving and loading Q-tables."""

    def test_save_and_load_q_table(self):
        """Test saving and loading Q-table."""
        agent = QLearningAgent()

        # Set some Q-values
        agent.q_table[0, :] = [1.0, 2.0, 3.0, 4.0]
        agent.q_table[5, :] = [10.0, 20.0, 30.0, 40.0]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_qtable.pkl"

            # Save
            agent.save_q_table(str(save_path))

            # Create new agent and load
            new_agent = QLearningAgent()
            new_agent.load_q_table(str(save_path))

            # Q-tables should match
            np.testing.assert_array_equal(agent.q_table, new_agent.q_table)

    def test_save_creates_directory(self):
        """Test that save creates directory if it doesn't exist."""
        agent = QLearningAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "subdir" / "test_qtable.pkl"

            # Directory doesn't exist yet
            assert not save_path.parent.exists()

            # Save should create it
            agent.save_q_table(str(save_path))

            assert save_path.exists()

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        agent = QLearningAgent()

        with pytest.raises(FileNotFoundError):
            agent.load_q_table("nonexistent_file.pkl")


class TestTrainingHistory:
    """Tests for training history tracking."""

    def test_history_initialization(self):
        """Test that training history is initialized."""
        agent = QLearningAgent()

        assert hasattr(agent, "training_history")
        assert isinstance(agent.training_history, list)
        assert len(agent.training_history) == 0

    def test_record_episode(self):
        """Test recording episode statistics."""
        agent = QLearningAgent()

        # Record an episode
        agent.record_episode(episode=1, score=25, epsilon=0.2, avg_q_value=5.0)

        assert len(agent.training_history) == 1
        record = agent.training_history[0]
        assert record["episode"] == 1
        assert record["score"] == 25
        assert record["epsilon"] == 0.2
        assert record["avg_q_value"] == 5.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_equal_q_values(self):
        """Test action selection when all Q-values are equal."""
        agent = QLearningAgent(epsilon=0.0)

        # All Q-values equal
        agent.q_table[0, :] = [5.0, 5.0, 5.0, 5.0]

        # Should still select a valid action
        action = agent.select_action(0)
        assert 0 <= action < 4

    def test_negative_q_values(self):
        """Test that negative Q-values are handled correctly."""
        agent = QLearningAgent(learning_rate=1.0, gamma=0.9)

        agent.q_table[0, :] = [-10.0, -5.0, -2.0, -8.0]

        # Should select action 2 (least negative)
        action = agent.select_action(0, training=False)
        assert action == 2

    def test_large_reward_values(self):
        """Test handling of large reward values."""
        agent = QLearningAgent(learning_rate=1.0, gamma=0.9)

        agent.update_q_value(0, 0, 1000.0, 1)

        assert agent.q_table[0, 0] == pytest.approx(1000.0)

    def test_zero_learning_rate(self):
        """Test that zero learning rate prevents updates."""
        agent = QLearningAgent(learning_rate=0.0)

        initial_value = agent.q_table[0, 0]
        agent.update_q_value(0, 0, 100.0, 1)

        # Q-value shouldn't change with learning_rate=0
        assert agent.q_table[0, 0] == initial_value
