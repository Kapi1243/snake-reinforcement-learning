# Contributing to Snake Q-Learning

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- Your environment (OS, Python version)
- Relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- Clear description of the proposed feature
- Rationale for why it would be useful
- Potential implementation approach (if applicable)

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/snake-reinforcement-learning.git
cd snake-reinforcement-learning

# Install dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Try the demo
python src/demo.py
```

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
