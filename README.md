# 🐍 Snake AI: Q-Learning Implementation

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A modern implementation of Q-Learning applied to the classic Snake game. This project demonstrates fundamental reinforcement learning concepts using a custom-built environment and tabular Q-learning agent.

<div align="center">
  <img src="Images/AnimatedGames.gif" alt="Snake AI in Action" width="600"/>
  <p><em>The agent learning to play Snake across different training episodes</em></p>
</div>

## 🎯 Project Overview

This project explores how an AI agent can learn to play Snake through **reinforcement learning** without any hardcoded game strategy. The agent learns purely from experience, using rewards and penalties to develop an optimal policy.

### Key Features

- ✅ **Custom Snake Environment**: Grid-based game implementation optimized for RL
- ✅ **Tabular Q-Learning**: Classic RL algorithm with configurable hyperparameters
- ✅ **State Representation**: Efficient 8-bit binary encoding (256 possible states)
- ✅ **Training Visualization**: Animated gameplay showing learning progress
- ✅ **Performance Analytics**: Convergence graphs and metrics tracking
- ✅ **Model Persistence**: Save and load trained models
- ✅ **Type Hints & Documentation**: Production-quality code standards

## 🧠 How It Works

### State Representation

The environment is encoded into an **8-dimensional binary state vector**:

| Bits 0-3 | Bits 4-7 |
|----------|----------|
| Obstacle detection (↑ → ↓ ←) | Food direction (↑ → ↓ ←) |

Each bit indicates:
- **Bits 0-3**: Whether moving in that direction would cause a collision (wall or body)
- **Bits 4-7**: Whether food is located in that general direction

This compact representation allows for 2^8 = 256 possible states, making tabular Q-learning feasible.

### Q-Learning Algorithm

The agent uses the **Q-learning update rule**:

```
Q(s, a) ← r + γ · max[Q(s', a')]
```

Where:
- `s` = current state
- `a` = action taken
- `r` = immediate reward
- `γ` = discount factor (default: 0.8)
- `s'` = next state
- `a'` = possible next actions

### Reward Structure

| Event | Reward |
|-------|--------|
| Eating food | +10 |
| Moving toward food | +1 |
| Collision (death) | -10 |

## 📊 Results

The agent achieves significant improvement through training:

<div align="center">
  <img src="Images/ConvergenceGraph.png" alt="Q-Table Convergence" width="600"/>
  <p><em>Q-value convergence over 5,000 training episodes</em></p>
</div>

### Performance Metrics

| Metric | Random Policy | Trained Agent |
|--------|---------------|---------------|
| Avg. Length | 2-3 | 15-20 |
| Max Length | 5 | 50+ |
| Success Rate* | 5% | 85% |

<sup>*Success rate = games reaching length ≥10</sup>

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Kapi1243/snake-reinforcement-learning.git
cd snake-reinforcement-learning
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Quick Start

#### Play the game yourself
```bash
python src/Snake.py
```
Controls: W (up), A (left), S (down), D (right), Q (quit)

#### Train a new agent
```bash
python src/QLearningAgent.py
```

#### Use pre-trained model
```python
from src.QLearningAgent import QLearningAgent

agent = QLearningAgent()
agent.load('models/q_learning_snake.pkl')

# Evaluate performance
avg_score, scores = agent.evaluate(board_size=16, num_episodes=100)
print(f"Average score: {avg_score}")
```

## 📁 Project Structure

```
snake-reinforcement-learning/
├── src/
│   ├── Snake.py              # Game environment
│   ├── QLearningAgent.py     # Q-Learning implementation
│   └── Visualizations/
│       └── makeQconvergenceGraph.py
├── Images/
│   ├── AnimatedGames.gif     # Training animation
│   └── ConvergenceGraph.png  # Q-value convergence plot
├── models/                   # Saved Q-tables
├── results/                  # Training outputs
├── requirements.txt
└── README.md
```

## 🎓 Learning Outcomes

This project demonstrates:

1. **Reinforcement Learning Fundamentals**
   - Markov Decision Processes (MDPs)
   - Q-learning algorithm
   - Exploration vs. exploitation trade-off

2. **Environment Design**
   - State space engineering
   - Reward shaping
   - Episode termination conditions

3. **Software Engineering**
   - Object-oriented design
   - Type hints and documentation
   - Model persistence
   - Visualization techniques

## 🔧 Hyperparameter Tuning

Key parameters and their effects:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gamma` (γ) | 0.8 | Higher → values future rewards more |
| `epsilon` (ε) | 0.2 | Higher → more exploration during training |
| `learning_rate` (α) | 0.9 | Higher → faster but less stable learning |
| `board_size` | 16×16 | Larger → harder but more interesting |

## 🎨 Customization

### Modify reward structure
```python
# In src/Snake.py, edit the step() method
def step(self, action: int) -> Tuple[int, float, bool, int]:
    # ... existing code ...
    if moving_toward_food:
        reward = 2.0  # Increase from 1.0
    # ... rest of method ...
```

### Change board size
```python
agent = QLearningAgent()
agent.train(board_size=20, num_episodes=10000)  # 20×20 board
```

### Adjust exploration rate
```python
agent = QLearningAgent(epsilon=0.3)  # More exploration
```

## 🔮 Future Improvements

Potential enhancements to explore:

- [ ] **Deep Q-Networks (DQN)**: Replace Q-table with neural network for larger state spaces
- [ ] **Double DQN**: Reduce overestimation bias
- [ ] **Dueling DQN**: Separate value and advantage streams
- [ ] **Prioritized Experience Replay**: Learn from important transitions more frequently
- [ ] **Multi-step returns**: Use n-step TD learning
- [ ] **Curriculum learning**: Gradually increase board size
- [ ] **Opponent snake**: Multi-agent competitive environment

## 📚 References

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) - Sutton & Barto
- [Q-Learning Tutorial](http://mnemstudio.org/path-finding-q-learning-tutorial.htm)
- [Simple Reinforcement Learning: Q-learning](https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 👤 Author

**Kacper Kowalski**
- GitHub: [@Kapi1243](https://github.com/Kapi1243)

## 🙏 Acknowledgments

- Inspiration from various Snake RL implementations in the open-source community
- The reinforcement learning community for excellent educational resources

---

<div align="center">
</div>
