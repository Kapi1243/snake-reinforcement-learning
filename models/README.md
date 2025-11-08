# Models Directory

This directory contains trained Q-learning models saved as `.pkl` files.

## Loading a Model

```python
from src.QLearningAgent import QLearningAgent

agent = QLearningAgent()
agent.load('models/q_learning_snake.pkl')

# Evaluate the loaded model
avg_score, scores = agent.evaluate(board_size=16, num_episodes=100)
print(f"Average score: {avg_score}")
```

## Training and Saving

```python
agent = QLearningAgent(gamma=0.9, epsilon=0.2, learning_rate=0.1)
agent.train(board_size=16, num_episodes=20000)
agent.save('models/my_model.pkl')
```
