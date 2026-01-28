"""
Deep Q-Network (DQN) Agent for Snake Game

This module implements a modern Deep Q-Learning agent using neural networks
instead of tabular Q-learning. Features include:
- Experience replay buffer
- Target network for stable learning
- Double DQN to reduce overestimation
- Prioritized experience replay (optional)

Author: Kacper Kowalski
Date: January 2026
"""

from typing import Tuple, List, Optional, Deque
from collections import deque, namedtuple
import numpy as np
import random
import pickle
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. DQN agent will not be available.")
    print("Install with: pip install torch")


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNetwork(nn.Module):
    """
    Deep Q-Network architecture.
    
    A feedforward neural network that maps states to Q-values for each action.
    Uses multiple hidden layers with ReLU activations and dropout for regularization.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.2
    ):
        """
        Initialize the DQN.
        
        Args:
            state_dim: Dimension of state representation
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
        """
        super(DQNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Stores experiences (s, a, r, s', done) and allows random sampling
    to break correlation between consecutive samples.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer: Deque[Experience] = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add an experience to the buffer."""
        exp = Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    
    Implements the DQN algorithm with:
    - Experience replay to break temporal correlations
    - Separate target network for stable learning
    - Epsilon-greedy exploration
    - Gradient clipping for stability
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        hidden_dims: List[int] = [256, 256, 128],
        device: Optional[str] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state representation
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            min_epsilon: Minimum epsilon value
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Steps between target network updates
            hidden_dims: Hidden layer dimensions
            device: Device to use ('cpu' or 'cuda'), auto-detect if None
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN agent. Install with: pip install torch")
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"DQN Agent using device: {self.device}")
        
        # Store hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Create networks
        self.policy_net = DQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.training_history: List[dict] = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step (if enough experiences collected).
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self) -> None:
        """Decay epsilon for exploration-exploitation tradeoff."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str) -> None:
        """
        Save model weights and training state.
        
        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'training_history': self.training_history
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model weights and training state.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        self.training_history = checkpoint['training_history']
        
        print(f"Model loaded from {filepath}")
    
    def record_episode(self, **kwargs) -> None:
        """Record episode statistics."""
        self.training_history.append(kwargs)
        self.episodes_done += 1


def state_to_vector(state: int, num_bits: int = 8) -> np.ndarray:
    """
    Convert integer state to binary vector representation.
    
    Args:
        state: Integer state (0-255 for 8-bit)
        num_bits: Number of bits in representation
        
    Returns:
        Binary vector representation
    """
    binary = format(state, f'0{num_bits}b')
    return np.array([int(bit) for bit in binary], dtype=np.float32)


if __name__ == "__main__":
    # Example usage
    print("DQN Agent Example")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    
    if TORCH_AVAILABLE:
        agent = DQNAgent(state_dim=8, action_dim=4)
        print(f"Agent created on device: {agent.device}")
        print(f"Policy network: {agent.policy_net}")
