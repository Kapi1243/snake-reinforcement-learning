# Quick Start Guide

Get up and running with Snake Q-Learning in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/Kapi1243/snake-reinforcement-learning.git
cd snake-reinforcement-learning

# Install dependencies
pip install -r requirements.txt
```

## Option 1: Interactive Demo (Recommended)

```bash
python src/demo.py
```

This launches an interactive menu where you can:
1. Play Snake yourself
2. Watch quick training
3. See agent evaluation
4. Watch the agent play

## Option 2: Jupyter Notebook

```bash
jupyter notebook notebooks/Snake_QLearning_Tutorial.ipynb
```

Follow the step-by-step tutorial with visualizations.

## Option 3: Train Your Own Agent

```python
from src.QLearningAgent import QLearningAgent

# Create agent
agent = QLearningAgent(gamma=0.8, epsilon=0.2)

# Train
agent.train(
    board_size=16,
    num_episodes=5000,
    verbose=True
)

# Evaluate
avg_score, scores = agent.evaluate(board_size=16, num_episodes=100)
print(f"Average score: {avg_score}")

# Save
agent.save('models/my_agent.pkl')

# Visualize
agent.plot_training_progress()
```

## Option 4: Play the Game

```bash
python src/Snake.py
```

Controls: W (up), A (left), S (down), D (right), Q (quit)

## What's Next?

- Read the [full README](README.md) for detailed documentation
- Explore [configuration options](src/config.py)
- Check out [utility functions](src/utils.py)
- Experiment with different hyperparameters
- Try the advanced features in the notebook

## Troubleshooting

**Import errors?**
```bash
pip install --upgrade numpy matplotlib pillow
```

**Visualization not showing?**
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

**Training too slow?**
- Reduce `num_episodes` (try 1000 for quick test)
- Use smaller `board_size` (8x8 instead of 16x16)
- Disable `save_q_history`

## Need Help?

- Check [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
- Open an issue on GitHub
- Review the [notebook tutorial](notebooks/Snake_QLearning_Tutorial.ipynb)

Happy learning! 🐍
