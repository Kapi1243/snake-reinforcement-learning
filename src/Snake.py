"""
Snake Game Environment for Reinforcement Learning

A custom implementation of the classic Snake game designed specifically for 
Q-learning experiments. Features a grid-based environment with state 
representation optimized for reinforcement learning algorithms.

Author: Kacper Kowalski
Last Update Date: 01 November 2025
"""

from typing import Tuple, Optional, List
import numpy as np
import random


class BodyNode:
    """
    Represents a single segment of the snake's body.
    
    Uses a linked-list structure where each node points to its parent
    (the segment closer to the head), enabling efficient body movement.
    
    Attributes:
        parent (BodyNode): Reference to the next segment toward the head
        x (int): Horizontal position on the grid
        y (int): Vertical position on the grid
    """
    
    def __init__(self, parent: Optional['BodyNode'], x: int, y: int) -> None:
        """
        Initialize a body segment.
        
        Args:
            parent: The parent node (closer to head), None if this is the head
            x: X-coordinate on the game grid
            y: Y-coordinate on the game grid
        """
        self.parent = parent
        self.x = x
        self.y = y

    def set_position(self, x: int, y: int) -> None:
        """Update the position of this body segment."""
        self.x = x
        self.y = y

    def set_parent(self, parent: 'BodyNode') -> None:
        """Update the parent reference."""
        self.parent = parent

    def get_position(self) -> Tuple[int, int]:
        """Return position as (x, y) tuple."""
        return (self.x, self.y)
    
    def get_index(self) -> Tuple[int, int]:
        """Return position as (y, x) tuple for array indexing."""
        return (self.y, self.x)


class Snake:
    """
    Snake entity using a linked-list structure for body segments.
    
    The snake moves by updating each body segment to follow the position
    of its parent segment, creating smooth slithering movement.
    """
    
    # Direction constants for clarity
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(self, x: int, y: int) -> None:
        """
        Initialize snake with a single segment at the given position.
        
        Args:
            x: Starting x-coordinate
            y: Starting y-coordinate
        """
        self.head = BodyNode(None, x, y)
        self.tail = self.head

    def _move_body_forward(self) -> None:
        """
        Propagate movement through the body segments.
        
        Each segment moves to its parent's previous position, creating
        the characteristic snake movement pattern.
        """
        current_node = self.tail
        while current_node.parent is not None:
            parent_position = current_node.parent.get_position()
            current_node.set_position(*parent_position)
            current_node = current_node.parent

    def move(self, direction: int) -> Tuple[int, int, int, int]:
        """
        Move the snake in the specified direction.
        
        Args:
            direction: Movement direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            
        Returns:
            Tuple of (old_tail_x, old_tail_y, new_head_x, new_head_y)
        """
        old_tail_x, old_tail_y = self.tail.get_position()
        self._move_body_forward()
        
        head_x, head_y = self.head.get_position()
        
        # Update head position based on direction
        if direction == self.UP:
            head_y -= 1
        elif direction == self.RIGHT:
            head_x += 1
        elif direction == self.DOWN:
            head_y += 1
        elif direction == self.LEFT:
            head_x -= 1
            
        self.head.set_position(head_x, head_y)
        return (old_tail_x, old_tail_y, *self.head.get_position())

    def grow(self, new_x: int, new_y: int) -> None:
        """
        Add a new segment at the head position (when food is eaten).
        
        Args:
            new_x: X-coordinate for new head position
            new_y: Y-coordinate for new head position
        """
        new_head = BodyNode(None, new_x, new_y)
        self.head.set_parent(new_head)
        self.head = new_head
        
    def get_head(self) -> BodyNode:
        """Return the head segment."""
        return self.head
    
    def get_tail(self) -> BodyNode:
        """Return the tail segment."""
        return self.tail


class SnakeGame:
    """
    Snake game environment designed for reinforcement learning.
    
    This environment provides:
    - State representation as an 8-bit binary vector
    - Reward signals for learning (positive for food, negative for collisions)
    - Game board visualization for monitoring training
    
    The state encoding captures:
    - Obstacle detection in 4 directions (walls/body)
    - Food location relative to head (in 1-2 cardinal directions)
    
    Attributes:
        width (int): Game board width
        height (int): Game board height
        board (np.ndarray): 2D array representing the game state
        snake (Snake): The snake entity
        length (int): Current snake length (score)
    """
    
    # Cell type constants
    EMPTY = 0
    BODY = 1
    HEAD = 2
    FOOD = 7
    
    def __init__(self, width: int = 16, height: int = 16) -> None:
        """
        Initialize a new game instance.
        
        Args:
            width: Board width (default: 16)
            height: Board height (default: 16)
        """
        self.width = width
        self.height = height
        self.board = np.zeros([height, width], dtype=int)
        self.length = 1

        # Spawn snake at center
        start_x = width // 2
        start_y = height // 2

        self.board[start_y, start_x] = self.HEAD
        self.snake = Snake(start_x, start_y)
        
        self._spawn_food()
        self._calculate_state()

    def _spawn_food(self) -> None:
        """
        Spawn food at a random empty position.
        
        Searches for all empty cells and randomly selects one for food placement.
        """
        empty_cells = []
        for index, value in np.ndenumerate(self.board):
            if value == self.EMPTY:
                empty_cells.append(index)
        
        if empty_cells:
            self.food_index = random.choice(empty_cells)
            self.board[self.food_index] = self.FOOD

    def _is_valid_move(self, direction: int) -> bool:
        """
        Check if a move in the given direction is valid.
        
        Args:
            direction: Direction to check (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            
        Returns:
            True if the move is valid (no wall or self-collision), False otherwise
        """
        new_x, new_y = self._get_potential_position(direction)
        
        # Check wall collision
        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
            return False
        
        # Check self-collision
        if self.board[new_y, new_x] == self.BODY:
            return False
        
        return True

    def _get_potential_position(self, direction: int) -> Tuple[int, int]:
        """
        Calculate the position that would result from moving in the given direction.
        
        Args:
            direction: Direction to move (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            
        Returns:
            Tuple of (x, y) coordinates
        """
        new_x, new_y = self.snake.get_head().get_position()
        
        if direction == Snake.UP:
            new_y -= 1
        elif direction == Snake.RIGHT:
            new_x += 1
        elif direction == Snake.DOWN:
            new_y += 1
        elif direction == Snake.LEFT:
            new_x -= 1
        
        return (new_x, new_y)

    def _calculate_state(self) -> None:
        """
        Compute the current state representation.
        
        State is an 8-element binary vector:
        - Elements 0-3: Obstacle (wall/body) in each direction (UP, RIGHT, DOWN, LEFT)
        - Elements 4-7: Food presence in each direction
        """
        self.state = np.zeros(8, dtype=int)
        
        # Check for obstacles in each direction
        for direction in range(4):
            self.state[direction] = not self._is_valid_move(direction)
        
        # Encode food direction
        self.state[4:] = self._get_food_direction()

    def _get_state_number(self) -> int:
        """
        Convert the binary state vector to a unique integer.
        
        This creates a unique state ID from 0 to 255 (2^8 - 1) for Q-table indexing.
        
        Returns:
            Integer representation of the current state
        """
        state_num = 0
        for i in range(8):
            state_num += 2**i * self.state[i]
        return state_num

    def _get_food_direction(self) -> np.ndarray:
        """
        Determine which cardinal directions contain the food.
        
        Food can be in 1-2 directions (e.g., "up and right" or just "left").
        
        Returns:
            4-element binary array indicating food presence in each direction
        """
        food_directions = np.zeros(4, dtype=int)
        
        # Calculate distance vector from head to food
        head_index = self.snake.get_head().get_index()
        distance = np.array(self.food_index) - np.array(head_index)
        
        # Vertical direction
        if distance[0] < 0:
            food_directions[0] = 1  # Up
        elif distance[0] > 0:
            food_directions[2] = 1  # Down
        
        # Horizontal direction
        if distance[1] > 0:
            food_directions[1] = 1  # Right
        elif distance[1] < 0:
            food_directions[3] = 1  # Left
        
        return food_directions

    def get_plottable_board(self) -> np.ndarray:
        """
        Generate a visualization-friendly representation of the board.
        
        Returns a board where:
        - Food is represented as -1
        - Snake body segments have gradient values (0.2 to 1.0) from tail to head
        
        Returns:
            2D numpy array suitable for plotting
        """
        board = np.zeros([self.height, self.width])
        
        # Draw snake with gradient from tail to head
        current_node = self.snake.tail
        segment_count = 0
        
        while current_node is not None:
            segment_count += 1
            intensity = 0.2 + 0.8 * segment_count / self.length
            board[current_node.get_index()] = intensity
            current_node = current_node.parent
        
        # Draw food
        board[self.food_index] = -1
        
        return board
        
        
    def display(self) -> None:
        """Print a text-based representation of the game board to console."""
        # Top border
        print('-' * (self.width + 2))
        
        # Board content
        for i in range(self.height):
            print('|', end='')
            for j in range(self.width):
                cell = self.board[i, j]
                if cell == self.EMPTY:
                    print(' ', end='')
                elif cell == self.HEAD:
                    print('O', end='')
                elif cell == self.BODY:
                    print('X', end='')
                elif cell == self.FOOD:
                    print('*', end='')
            print('|')
        
        # Bottom border
        print('-' * (self.width + 2))

    def step(self, action: int) -> Tuple[int, float, bool, int]:
        """
        Execute one step in the environment.
        
        This is the main interface for reinforcement learning agents.
        
        Args:
            action: Direction to move (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            
        Returns:
            Tuple containing:
                - state (int): New state number
                - reward (float): Reward for this action
                - done (bool): Whether the game is over
                - score (int): Current snake length
        """
        game_over = False
        reward = 0.0
        
        if self._is_valid_move(action):
            # Small survival bonus to encourage longer games
            reward = 0.1
            
            # Additional reward for moving toward food
            if self._get_food_direction()[action] == 1:
                reward += 1.0
            
            # Update old head position to body
            head_x, head_y = self.snake.get_head().get_position()
            self.board[head_y, head_x] = self.BODY

            # Check if moving to food
            pot_x, pot_y = self._get_potential_position(action)
            
            if self.board[pot_y, pot_x] == self.FOOD:
                # Snake eats food and grows
                self.snake.grow(pot_x, pot_y)
                self.board[pot_y, pot_x] = self.HEAD
                self._spawn_food()
                self.length += 1
                # Scaled reward: longer snake = higher reward for continued success
                reward = 10.0 + (self.length * 0.5)
            else:
                # Normal movement
                old_tail_x, old_tail_y, new_head_x, new_head_y = self.snake.move(action)
                self.board[old_tail_y, old_tail_x] = self.EMPTY
                self.board[new_head_y, new_head_x] = self.HEAD
        else:
            # Invalid move (collision)
            reward = -10.0
            game_over = True
        
        self._calculate_state()
        return (self._get_state_number(), reward, game_over, self.length)

    def reset(self) -> int:
        """
        Reset the game to initial state.
        
        Returns:
            Initial state number
        """
        self.__init__(self.width, self.height)
        return self._get_state_number()

    def get_state(self) -> np.ndarray:
        """Return the current state vector."""
        return self.state.copy()


def play_human_game(board_size: int = 8) -> None:
    """
    Run an interactive game for human players.
    
    Args:
        board_size: Size of the game board (default: 8x8)
    """
    game = SnakeGame(board_size, board_size)
    game.display()
    print(f"Score: {game.length}")
    print("\nControls: W=Up, A=Left, S=Down, D=Right, Q=Quit\n")
    
    direction_map = {
        'w': Snake.UP,
        'a': Snake.LEFT,
        's': Snake.DOWN,
        'd': Snake.RIGHT
    }
    
    while True:
        user_input = input("Direction: ").lower()
        
        if user_input == 'q':
            print("Game quit by user.")
            break
        
        if user_input not in direction_map:
            print("Invalid input! Use W, A, S, D, or Q")
            continue
        
        direction = direction_map[user_input]
        new_state, reward, game_over, score = game.step(direction)
        
        if game_over:
            game.display()
            print(f"\n=== GAME OVER ===")
            print(f"Final Score: {score}")
            print(f"Reward: {reward}")
            break
        else:
            game.display()
            print(f"Score: {score} | Reward: {reward}")


if __name__ == "__main__":
    play_human_game()
