# Snake AI: Reinforcement Learning Implementation

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Production-ready reinforcement learning framework featuring tabular Q-Learning and Deep Q-Networks (DQN) applied to the classic Snake game. Demonstrates professional software engineering practices including comprehensive testing, CI/CD, and experiment tracking.

## Quick Start

```bash
# Install
git clone https://github.com/Kapi1243/snake-reinforcement-learning.git
cd snake-reinforcement-learning
pip install -e .

# Run
python play.py
```

This will train an AI to play Snake and display results with statistics.

## How It Works

### State Representation

The environment is encoded into an 8-bit state vector:

| Bits 0-3 | Bits 4-7 |
|----------|----------|
| Obstacle detection (↑ → ↓ ←) | Food direction (↑ → ↓ ←) |

This gives 256 possible states for efficient tabular Q-learning.

### Q-Learning Algorithm

The agent learns using the incremental Q-learning update rule:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max[Q(s', a')] - Q(s, a)]
```

Where:
- `α` = learning rate (default: 0.1)
- `γ` = discount factor (default: 0.9)
- `r` = reward signal
- `s` = state, `a` = action

### Reward Structure

| Event | Reward |
|-------|--------|
| Eating food | +10.0 + (length × 0.5) |
| Moving toward food | +1.0 |
| Survival (per step) | +0.1 |
| Collision | -10.0 |

## Project Structure

```
src/
├── Snake.py                 # Game environment
├── QLearningAgent.py        # Q-Learning agent
├── DQNAgent.py              # Deep Q-Network (PyTorch)
├── cli.py                   # Command-line interface
├── config.py                # Configuration presets
├── experiment_tracking.py   # Metrics logging
├── profiling.py             # Performance tools
└── utils.py                 # Utilities

tests/                       # 270+ unit tests
.github/workflows/          # CI/CD pipelines
notebooks/                  # Jupyter tutorials
models/                     # Saved trained models
results/                    # Training outputs
```

## Features

**Algorithms**
- Tabular Q-Learning with epsilon-decay
- Deep Q-Network (DQN) with experience replay
- Efficient state encoding (8 bits = 256 states)
- Reward shaping for accelerated learning

**Software Quality**
- 270+ unit tests (85%+ code coverage)
- Type hints throughout (mypy validated)
- CI/CD pipeline (GitHub Actions)
- Pre-commit hooks for code quality
- Black, isort, flake8, mypy, bandit

**Experiment Tracking**
- TensorBoard integration
- Weights & Biases support
- JSON logging
- Multi-seed evaluation framework

## Usage

### Simple Training

```python
from src.QLearningAgent import QLearningAgent

agent = QLearningAgent(gamma=0.9, learning_rate=0.1, epsilon=0.3)
results = agent.train(board_size=12, num_episodes=5000)
agent.plot_training_progress()
```

### Custom Configuration

```python
from src.Snake import SnakeGame
from src.QLearningAgent import QLearningAgent

game = SnakeGame(width=16, height=16)
agent = QLearningAgent(gamma=0.95, learning_rate=0.3, epsilon=0.3)

for episode in range(5000):
    state = game.reset()
    while True:
        action = agent.select_action(state, training=True)
        next_state, reward, done, score = game.step(action)
        agent.update_q_value(state, action, reward, next_state)
        if done:
            break
        state = next_state
    agent.decay_epsilon()
```

### Evaluation

```python
avg_score, scores = agent.evaluate(board_size=12, num_episodes=100)
print(f"Average: {avg_score:.2f}")
print(f"Best: {max(scores)}")
print(f"Fail rate (<5): {sum(1 for s in scores if s < 5) / 100 * 100:.1f}%")
```

## Development

```bash
# Run tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Code quality
black src tests
isort src tests
mypy src
flake8 src

# Pre-commit setup
pre-commit install
```

## Current Optimal Hyperparameters

Based on 5-seed evaluation:

| Parameter | Value | Notes |
|-----------|-------|-------|
| gamma | 0.9 | Discount factor |
| learning_rate | 0.3 | Faster convergence |
| epsilon | 0.3 | 30% exploration |
| board_size | 12 | Balance difficulty/learnability |
| num_episodes | 5000 | Training budget |

Average performance: **26.45 ± 1.44** | Best: 52 | Fail rate: 0%

## References

## License

MIT - See [LICENSE](LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Author

Kacper Kowalski - [@Kapi1243](https://github.com/Kapi1243)
