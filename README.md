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
- ✅ **Tabular Q-Learning**: Incremental updates with epsilon decay
- ✅ **State Representation**: Efficient 8-bit binary encoding (256 possible states)
- ✅ **Length-Based Rewards**: Scaled rewards encouraging longer survival
- ✅ **Early Stopping**: Automatic training termination when plateaued
- ✅ **Training Visualization**: Auto-generated performance curves
- ✅ **Model Persistence**: Save and load trained Q-tables
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

The agent uses the **incremental Q-learning update rule**:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max[Q(s', a')] - Q(s, a)]
```

Where:
- `s` = current state
- `a` = action taken
- `r` = immediate reward
- `α` = learning rate (default: 0.1)
- `γ` = discount factor (default: 0.9)
- `s'` = next state
- `a'` = possible next actions

### Reward Structure

| Event | Reward |
|-------|--------|
| Eating food | +10.0 + (length × 0.5) |
| Moving toward food | +1.1 |
| Survival (each step) | +0.1 |
| Collision (death) | -10 |

## 📊 Results

The agent achieves consistent performance through Q-learning with epsilon decay and length-based reward scaling:

<div align="center">
  <img src="results/training_progress.png" alt="Training Progress" width="700"/>
  <p><em>Training progress showing average and maximum scores over episodes</em></p>
</div>

### Performance Metrics

| Metric | Value |
|--------|-------|
| Best Average Score | 33.20 |
| Best Single Score | 62 |
| Board Coverage | 24.2% (62/256 squares) |
| Training Episodes | ~15,700 (early stopping) |

The agent demonstrates solid learning on a 16×16 board using only 256 discrete states, showing the effectiveness of well-designed state representation and reward shaping.

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
│   ├── QLearningAgent.py     # Q-Learning agent (main implementation)
│   ├── config.py             # Configuration management
│   ├── utils.py              # Visualization utilities
│   ├── demo.py               # Interactive demos
│   └── Visualizations/
│       └── makeQconvergenceGraph.py
├── notebooks/
│   └── Snake_QLearning_Tutorial.ipynb  # Jupyter tutorial
├── Images/
│   ├── AnimatedGames.gif     # Training animation
│   └── ConvergenceGraph.png  # Q-value convergence plot
├── models/                   # Saved Q-tables
├── results/                  # Training outputs
├── requirements.txt
├── setup.py
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
| `gamma` (γ) | 0.9 | Higher → values future rewards more |
| `epsilon` (ε) | 0.2 → 0.05 | Decays during training for exploration→exploitation |
| `epsilon_decay` | 0.9995 | Controls rate of epsilon reduction |
| `learning_rate` (α) | 0.1 | Higher → faster but less stable learning |
| `board_size` | 16×16 | Larger → harder but more realistic |

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
agent = QLearningAgent(
    epsilon=0.3,           # Higher initial exploration
    epsilon_decay=0.999,   # Faster decay
    min_epsilon=0.01       # Lower minimum
)
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
