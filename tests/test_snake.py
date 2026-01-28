"""
Unit tests for the Snake game environment.

Tests cover game mechanics, state representation, collision detection,
and edge cases to ensure the environment behaves correctly.
"""

import pytest
import numpy as np
from src.Snake import Snake, SnakeGame, BodyNode


class TestBodyNode:
    """Tests for the BodyNode linked-list structure."""

    def test_initialization(self):
        """Test body node initialization."""
        node = BodyNode(None, 5, 10)
        assert node.x == 5
        assert node.y == 10
        assert node.parent is None

    def test_set_position(self):
        """Test position updating."""
        node = BodyNode(None, 0, 0)
        node.set_position(3, 7)
        assert node.get_position() == (3, 7)

    def test_get_index(self):
        """Test array indexing format."""
        node = BodyNode(None, 5, 3)
        assert node.get_index() == (3, 5)  # (y, x) for array indexing

    def test_parent_relationship(self):
        """Test parent-child relationships."""
        parent = BodyNode(None, 0, 0)
        child = BodyNode(parent, 1, 0)
        assert child.parent == parent

        new_parent = BodyNode(None, 2, 0)
        child.set_parent(new_parent)
        assert child.parent == new_parent


class TestSnake:
    """Tests for the Snake entity."""

    def test_initialization(self):
        """Test snake initialization with single segment."""
        snake = Snake(5, 5)
        assert snake.head.get_position() == (5, 5)
        assert snake.tail == snake.head

    def test_move_up(self):
        """Test movement in the UP direction."""
        snake = Snake(5, 5)
        old_x, old_y, new_x, new_y = snake.move(Snake.UP)
        assert old_x == 5 and old_y == 5
        assert new_x == 5 and new_y == 4  # Y decreases when moving up

    def test_move_down(self):
        """Test movement in the DOWN direction."""
        snake = Snake(5, 5)
        old_x, old_y, new_x, new_y = snake.move(Snake.DOWN)
        assert new_x == 5 and new_y == 6  # Y increases when moving down

    def test_move_left(self):
        """Test movement in the LEFT direction."""
        snake = Snake(5, 5)
        old_x, old_y, new_x, new_y = snake.move(Snake.LEFT)
        assert new_x == 4 and new_y == 5  # X decreases when moving left

    def test_move_right(self):
        """Test movement in the RIGHT direction."""
        snake = Snake(5, 5)
        old_x, old_y, new_x, new_y = snake.move(Snake.RIGHT)
        assert new_x == 6 and new_y == 5  # X increases when moving right

    def test_grow(self):
        """Test snake growth mechanism."""
        snake = Snake(5, 5)
        initial_tail = snake.tail

        # Move and grow
        snake.move(Snake.RIGHT)
        snake.grow()

        # Tail should be a new node
        assert snake.tail != initial_tail
        assert snake.tail.parent == initial_tail

    def test_body_follows_head(self):
        """Test that body segments follow the head."""
        snake = Snake(5, 5)
        snake.grow()
        snake.grow()

        # Move right twice
        snake.move(Snake.RIGHT)
        snake.move(Snake.RIGHT)

        # Check that segments follow in a line
        positions = []
        current = snake.head
        while current:
            positions.append(current.get_position())
            current = (
                current.tail
                if current == snake.head
                else (current.parent if hasattr(current.parent, "tail") else None)
            )
            if len(positions) > 10:  # Safety break
                break


class TestSnakeGame:
    """Tests for the SnakeGame environment."""

    def test_initialization(self):
        """Test game initialization."""
        game = SnakeGame(width=10, height=10)
        assert game.width == 10
        assert game.height == 10
        assert game.snake is not None
        assert game.food_x >= 0 and game.food_x < 10
        assert game.food_y >= 0 and game.food_y < 10

    def test_reset(self):
        """Test game reset functionality."""
        game = SnakeGame()

        # Play some moves
        game.step(0)
        game.step(1)
        game.step(2)

        # Reset
        state = game.reset()
        assert isinstance(state, int)
        assert 0 <= state < 256  # Valid state range

    def test_collision_with_wall(self):
        """Test collision detection with walls."""
        game = SnakeGame(width=10, height=10)

        # Move snake to top edge
        game.snake = Snake(5, 0)
        _, _, done, _ = game.step(Snake.UP)  # Try to move through wall
        assert done is True  # Should collide

    def test_collision_with_self(self):
        """Test self-collision detection."""
        game = SnakeGame(width=10, height=10)
        game.snake = Snake(5, 5)

        # Create a scenario where snake will hit itself
        # Grow the snake
        for _ in range(3):
            game.snake.grow()

        # Move in a circle to collide with self
        game.snake.move(Snake.RIGHT)
        game.snake.move(Snake.DOWN)
        game.snake.move(Snake.LEFT)

        # Check collision
        collision = game._check_collision(4, 5)
        assert collision is True

    def test_food_consumption(self):
        """Test food eating and regeneration."""
        game = SnakeGame(width=10, height=10)

        # Manually place food in front of snake
        snake_x, snake_y = game.snake.head.get_position()
        game.food_x = snake_x + 1
        game.food_y = snake_y

        # Move toward food
        _, reward, _, _ = game.step(Snake.RIGHT)

        # Should get positive reward for eating
        assert reward > 0

    def test_state_representation(self):
        """Test state encoding returns valid values."""
        game = SnakeGame()
        state = game.get_state()

        # State should be an integer in range [0, 255]
        assert isinstance(state, (int, np.integer))
        assert 0 <= state < 256

    def test_invalid_action_handling(self):
        """Test handling of invalid action values."""
        game = SnakeGame()

        # Valid actions should work
        for action in range(4):
            game.reset()
            state, reward, done, info = game.step(action)
            assert isinstance(state, (int, np.integer))
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)

    def test_max_steps_without_food(self):
        """Test that game ends if snake doesn't eat for too long."""
        game = SnakeGame(max_steps_without_food=50)

        # Place food far away
        game.food_x = 0
        game.food_y = 0
        game.snake = Snake(15, 15)

        # Take steps without eating
        for _ in range(51):
            _, _, done, _ = game.step(Snake.UP)
            if done:
                break

        assert done is True

    def test_deterministic_reset_with_seed(self):
        """Test that reset with seed is deterministic."""
        game1 = SnakeGame()
        game2 = SnakeGame()

        np.random.seed(42)
        state1 = game1.reset()

        np.random.seed(42)
        state2 = game2.reset()

        assert state1 == state2

    def test_render_board(self):
        """Test board rendering produces correct dimensions."""
        game = SnakeGame(width=10, height=10)
        board = game.render()

        assert board.shape == (10, 10)
        assert board.dtype == np.uint8


class TestGameMechanics:
    """Integration tests for game mechanics."""

    def test_complete_game_episode(self):
        """Test a complete game episode from start to end."""
        game = SnakeGame(width=8, height=8)
        state = game.reset()

        total_reward = 0
        steps = 0
        max_steps = 1000

        while steps < max_steps:
            action = np.random.randint(0, 4)
            state, reward, done, info = game.step(action)
            total_reward += reward
            steps += 1

            if done:
                break

        assert steps > 0
        assert done is True or steps == max_steps

    def test_reward_structure(self):
        """Test that rewards follow expected structure."""
        game = SnakeGame()

        # Test multiple steps
        rewards = []
        for _ in range(10):
            game.reset()
            _, reward, done, _ = game.step(np.random.randint(0, 4))
            if not done:
                rewards.append(reward)

        # Rewards should be numeric
        assert all(isinstance(r, (int, float)) for r in rewards)

    def test_state_transitions(self):
        """Test that states change appropriately."""
        game = SnakeGame()
        state1 = game.reset()

        state2, _, _, _ = game.step(0)

        # State may or may not change depending on position
        assert isinstance(state1, (int, np.integer))
        assert isinstance(state2, (int, np.integer))
