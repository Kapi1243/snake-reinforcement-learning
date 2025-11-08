"""
Q-Learning Agent for Snake Game

This module implements a tabular Q-Learning agent that learns to play Snake
through trial and error. The agent uses a Q-table to store state-action values
and employs epsilon-greedy exploration.

Features:
- Tabular Q-Learning with customizable hyperparameters
- Training progress monitoring and visualization
- Performance evaluation metrics
- Model persistence (save/load Q-tables)

Author: Kacper Kowalski
Date: November 2025
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
import random
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

from Snake import SnakeGame, Snake


class QLearningAgent:
    """
    Q-Learning agent for playing Snake.
    
    Uses tabular Q-learning to learn an optimal policy for the Snake game.
    The Q-table maps state-action pairs to expected cumulative rewards.
    
    Attributes:
        num_states (int): Number of possible states (2^8 = 256)
        num_actions (int): Number of possible actions (4 directions)
        q_table (np.ndarray): Q-value table of shape (num_states, num_actions)
        gamma (float): Discount factor for future rewards
        epsilon (float): Exploration rate for epsilon-greedy policy
        learning_rate (float): Learning rate (if using incremental updates)
    """
    
    def __init__(
        self,
        gamma: float = 0.8,
        epsilon: float = 0.2,
        learning_rate: float = 0.9,
        num_states: int = 256,
        num_actions: int = 4
    ):
        """
        Initialize the Q-Learning agent.
        
        Args:
            gamma: Discount factor (0-1), determines importance of future rewards
            epsilon: Exploration rate (0-1), probability of random action
            learning_rate: Learning rate for Q-value updates
            num_states: Number of possible states
            num_actions: Number of possible actions
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions))
        
        # Track training history
        self.training_history: List[Dict] = []
        self.q_history: List[np.ndarray] = []
    
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state number
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            Selected action (0-3)
        """
        if training and random.random() < self.epsilon:
            # Explore: choose random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: choose best known action
            return int(np.argmax(self.q_table[state, :]))
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update Q-value using the Q-learning update rule.
        
        Two update strategies are available:
        1. Direct replacement: Q(s,a) = r + γ * max(Q(s',a'))
        2. Incremental: Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        # Strategy 1: Direct replacement (simpler, works well for deterministic environments)
        self.q_table[state, action] = reward + self.gamma * np.max(self.q_table[next_state, :])
        
        # Strategy 2: Incremental update (uncomment to use)
        # current_q = self.q_table[state, action]
        # max_next_q = np.max(self.q_table[next_state, :])
        # self.q_table[state, action] = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
    
    def train(
        self,
        board_size: int = 16,
        num_episodes: int = 10000,
        eval_frequency: int = 100,
        eval_episodes: int = 25,
        save_q_history: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Train the agent on the Snake game.
        
        Args:
            board_size: Size of the game board
            num_episodes: Number of training episodes
            eval_frequency: How often to evaluate performance
            eval_episodes: Number of episodes for each evaluation
            save_q_history: Whether to save Q-table snapshots
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing training statistics
        """
        best_avg_score = 0
        best_q_table = None
        
        if verbose:
            print(f"Training Q-Learning Agent for {num_episodes} episodes...")
            print(f"Board size: {board_size}x{board_size}")
            print(f"Hyperparameters: γ={self.gamma}, ε={self.epsilon}, α={self.learning_rate}\n")
        
        for episode in range(num_episodes):
            game = SnakeGame(board_size, board_size)
            state = game._get_state_number()
            game_over = False
            
            # Play one episode
            while not game_over:
                action = self.select_action(state, training=True)
                next_state, reward, game_over, score = game.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
            
            # Periodic evaluation
            if episode % eval_frequency == 0:
                avg_score, scores = self.evaluate(board_size, eval_episodes)
                
                # Track best model
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_q_table = self.q_table.copy()
                
                # Record training progress
                self.training_history.append({
                    'episode': episode,
                    'avg_score': avg_score,
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'std_score': np.std(scores)
                })
                
                if verbose:
                    print(f"Episode {episode:5d} | Avg Score: {avg_score:.2f} | "
                          f"Best: {best_avg_score:.2f} | Range: [{min(scores)}-{max(scores)}]")
            
            # Save Q-table snapshot for convergence analysis
            if save_q_history and episode % 100 == 0:
                self.q_history.append(self.q_table.copy())
        
        # Restore best Q-table
        if best_q_table is not None:
            self.q_table = best_q_table
            if verbose:
                print(f"\nTraining complete! Best average score: {best_avg_score:.2f}")
        
        return {
            'best_avg_score': best_avg_score,
            'final_avg_score': self.training_history[-1]['avg_score'] if self.training_history else 0,
            'training_history': self.training_history
        }
    
    def evaluate(
        self,
        board_size: int = 16,
        num_episodes: int = 25,
        max_steps: int = 100
    ) -> Tuple[float, List[int]]:
        """
        Evaluate the agent's performance without exploration.
        
        Args:
            board_size: Size of the game board
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps without scoring before terminating
            
        Returns:
            Tuple of (average_score, list_of_scores)
        """
        scores = []
        
        for _ in range(num_episodes):
            game = SnakeGame(board_size, board_size)
            state = game._get_state_number()
            game_over = False
            steps_without_food = 0
            last_score = 1
            
            while not game_over:
                action = self.select_action(state, training=False)
                state, reward, game_over, score = game.step(action)
                
                # Detect infinite loops (snake going in circles)
                if score == last_score:
                    steps_without_food += 1
                else:
                    last_score = score
                    steps_without_food = 0
                
                if steps_without_food >= max_steps:
                    break
            
            scores.append(score)
        
        return np.mean(scores), scores
    
    def save(self, filepath: str) -> None:
        """
        Save the Q-table and agent parameters to disk.
        
        Args:
            filepath: Path to save the model
        """
        save_data = {
            'q_table': self.q_table,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a Q-table and agent parameters from disk.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.q_table = save_data['q_table']
        self.gamma = save_data['gamma']
        self.epsilon = save_data['epsilon']
        self.learning_rate = save_data['learning_rate']
        self.training_history = save_data.get('training_history', [])
        
        print(f"Model loaded from {filepath}")
        print(f"Trained at: {save_data.get('timestamp', 'Unknown')}")
    
    def plot_training_progress(self, save_path: Optional[str] = None) -> None:
        """
        Plot the training progress over time.
        
        Args:
            save_path: Optional path to save the figure
        """
        if not self.training_history:
            print("No training history available.")
            return
        
        episodes = [h['episode'] for h in self.training_history]
        avg_scores = [h['avg_score'] for h in self.training_history]
        min_scores = [h['min_score'] for h in self.training_history]
        max_scores = [h['max_score'] for h in self.training_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, avg_scores, 'b-', linewidth=2, label='Average Score')
        plt.fill_between(episodes, min_scores, max_scores, alpha=0.2, label='Min-Max Range')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Snake Length (Score)', fontsize=12)
        plt.title('Q-Learning Training Progress', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.show()


def create_training_animation(
    agent: QLearningAgent,
    board_size: int = 16,
    episodes_to_plot: List[int] = None,
    num_games: int = 10,
    max_frames: int = 1000,
    save_path: Optional[str] = None
) -> None:
    """
    Create an animated visualization of the agent's performance at different training stages.
    
    Args:
        agent: Trained Q-Learning agent with q_history
        board_size: Size of the game board
        episodes_to_plot: List of episode numbers to visualize
        num_games: Number of game runs to average over
        max_frames: Maximum number of frames per game
        save_path: Optional path to save the animation
    """
    if not agent.q_history:
        print("No Q-table history available. Train with save_q_history=True")
        return
    
    if episodes_to_plot is None:
        episodes_to_plot = [0, 200, 400, 600, 800, 1000, 2500, 5000, 10000]
    
    # Filter to available episodes
    episodes_to_plot = [ep for ep in episodes_to_plot if ep < len(agent.q_history) * 100]
    
    print("Generating animation data...")
    
    # Setup subplots
    rows = int(np.ceil(np.sqrt(len(episodes_to_plot))))
    cols = int(np.ceil(len(episodes_to_plot) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if len(episodes_to_plot) > 1 else [axes]
    
    # Initialize visualization components
    images = []
    labels = []
    frame_data = [[] for _ in episodes_to_plot]
    score_data = [[] for _ in episodes_to_plot]
    
    for idx, (ax, episode) in enumerate(zip(axes, episodes_to_plot)):
        ax.set_title(f"Episode {episode}", fontsize=10, fontweight='bold')
        ax.axis('off')
        
        im = ax.imshow(np.zeros([board_size, board_size]), vmin=-1, vmax=1, cmap='RdGy')
        label = ax.text(0, board_size - 1, 'Length: 1', 
                       bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3},
                       fontsize=8, verticalalignment='top')
        
        images.append(im)
        labels.append(label)
    
    # Hide extra subplots
    for idx in range(len(episodes_to_plot), len(axes)):
        axes[idx].axis('off')
    
    # Generate game data
    for game_num in range(num_games):
        games = []
        states = []
        game_overs = []
        cutoff_counters = []
        prev_scores = []
        
        # Initialize games for each episode
        for episode_idx, episode in enumerate(episodes_to_plot):
            game = SnakeGame(board_size, board_size)
            q_idx = min(episode // 100, len(agent.q_history) - 1)
            
            games.append(game)
            states.append(game._get_state_number())
            game_overs.append(False)
            cutoff_counters.append(0)
            prev_scores.append(1)
        
        # Run all games in parallel
        for frame in range(max_frames):
            for idx, episode in enumerate(episodes_to_plot):
                if game_overs[idx]:
                    # Repeat last frame if game over
                    if frame_data[idx]:
                        frame_data[idx].append(frame_data[idx][-1])
                        score_data[idx].append(score_data[idx][-1])
                    continue
                
                q_idx = min(episode // 100, len(agent.q_history) - 1)
                q_table = agent.q_history[q_idx]
                
                action = np.argmax(q_table[states[idx], :])
                states[idx], reward, game_over, score = games[idx].step(action)
                
                frame_data[idx].append(games[idx].get_plottable_board())
                score_data[idx].append(score)
                
                if game_over:
                    game_overs[idx] = True
                
                # Check for infinite loops
                if score == prev_scores[idx]:
                    cutoff_counters[idx] += 1
                else:
                    prev_scores[idx] = score
                    cutoff_counters[idx] = 0
                
                if cutoff_counters[idx] >= 100:
                    game_overs[idx] = True
            
            if all(game_overs):
                print(f"Game {game_num + 1}/{num_games} complete ({len(frame_data[0])} frames)")
                break
    
    # Animation function
    def animate(frame_num):
        for idx, (im, label) in enumerate(zip(images, labels)):
            if frame_num < len(frame_data[idx]):
                im.set_data(frame_data[idx][frame_num])
                label.set_text(f"Length: {score_data[idx][frame_num]}")
        return images + labels
    
    print("Creating animation...")
    num_frames = len(frame_data[0])
    ani = animation.FuncAnimation(
        fig, animate, frames=num_frames, 
        blit=True, interval=75, repeat=True
    )
    
    plt.tight_layout()
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, fps=15, writer='pillow')
        print(f"Animation saved!")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Snake Q-Learning Agent Training")
    print("="*60 + "\n")
    
    # Initialize agent
    agent = QLearningAgent(
        gamma=0.8,
        epsilon=0.2,
        learning_rate=0.9
    )
    
    # Train the agent
    training_results = agent.train(
        board_size=16,
        num_episodes=5000,
        eval_frequency=100,
        eval_episodes=25,
        save_q_history=False,  # Set to True for animation
        verbose=True
    )
    
    # Save the trained model
    agent.save('models/q_learning_snake.pkl')
    
    # Plot training progress
    agent.plot_training_progress(save_path='results/training_progress.png')
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    avg_score, scores = agent.evaluate(board_size=16, num_episodes=100)
    print(f"Average Score over 100 games: {avg_score:.2f}")
    print(f"Best Score: {max(scores)}")
    print(f"Score Range: [{min(scores)}, {max(scores)}]")
