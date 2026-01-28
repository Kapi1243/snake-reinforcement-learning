# Contributing to Snake Reinforcement Learning

Thank you for your interest in contributing!

## Quick Start

1. Fork the repository
2. Clone your fork
   ```bash
   git clone https://github.com/YOUR_USERNAME/snake-reinforcement-learning.git
   cd snake-reinforcement-learning
   ```

3. Set up development environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[dev]
   ```

4. Install pre-commit hooks
   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Style

Maintain code quality with:

- Black for code formatting (line length: 100)
- isort for import sorting
- flake8 for linting
- mypy for type checking
- bandit for security checks

Run all checks:
```bash
python scripts/check_code_quality.py
```

Or format code automatically:
```bash
black src tests
isort src tests
```

### Testing

Write tests for all new features:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_snake.py -v
```

### Type Hints

All new code should include type hints:

```python
def train_agent(
    agent: QLearningAgent,
    env: SnakeGame,
    episodes: int = 1000
) -> Dict[str, float]:
    """Train the agent."""
    ...
```

### Documentation

- Add docstrings to all public functions/classes (Google style)
- Update README.md for new features
- Add examples to docstrings when helpful

## Reporting Bugs

Include when reporting:

1. Python version and OS
2. Minimal code to reproduce
3. Expected vs actual behavior
4. Error messages/stack traces

## Feature Requests

For new features:

1. Check existing issues first
2. Describe the use case
3. Explain value proposition
4. Consider implementation approach

## Pull Request Process

1. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation

3. Commit your changes
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```
   
   Use conventional commits:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `test:` adding tests
   - `refactor:` code refactoring
   - `perf:` performance improvements

4. Run all checks
   ```bash
   python scripts/check_code_quality.py
   ```

5. Push and create PR
   ```bash
   git push origin feature/your-feature-name
   ```

## Project Structure

```
snake-reinforcement-learning/
├── src/                    # Source code
│   ├── Snake.py           # Game environment
│   ├── QLearningAgent.py  # Tabular Q-learning
│   ├── DQNAgent.py        # Deep Q-Network
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration management
│   ├── experiment_tracking.py  # Logging utilities
│   └── profiling.py       # Performance tools
├── tests/                 # Test suite
├── scripts/               # Utility scripts
├── models/                # Saved models
├── results/               # Training results
└── docs/                  # Documentation
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Add docstrings to all classes and functions
- Keep functions focused and under 50 lines when possible
- Use meaningful variable names

## Code Structure

```
src/
├── Snake.py           # Game environment
├── QLearningAgent.py  # Main agent implementation
├── config.py          # Configuration management
├── utils.py           # Utility functions
└── demo.py            # Demo script
```

## Testing

When adding new features, please:
- Test on different board sizes (8x8, 16x16, 24x24)
- Verify backward compatibility
- Check that models can be saved and loaded
- Ensure visualizations work correctly

## Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update notebook examples if relevant
- Comment complex algorithms

## Questions?

Feel free to open an issue with the "question" label!

Thank you for contributing! 🐍
