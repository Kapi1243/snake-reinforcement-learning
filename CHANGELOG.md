# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-11-08

### Added
- Epsilon decay mechanism (0.2 → 0.05) for better exploration-exploitation balance
- Length-based reward scaling (10.0 + length × 0.5) to encourage longer survival
- Small survival bonus (+0.1 per step) for safer gameplay
- Early stopping with patience=50 evaluations to save training time
- Auto-generated training curve visualization (`training_curve.png`)
- Best single score tracking during training
- Board coverage percentage in final metrics

### Changed
- Learning rate reduced from 0.9 to 0.1 for more stable updates
- Switched to incremental Q-value updates instead of direct replacement
- Gamma increased from 0.8 to 0.9 for better long-term planning
- Training now stops automatically when performance plateaus
- Enhanced training output showing epsilon decay and best scores

### Performance
- Best average score: 33.20
- Best single score: 62
- Board coverage: 24.2% (62/256 squares)
- Training episodes: ~15,700 (with early stopping)

## [2.0.0] - 2025-11-07

### Added
- Complete refactor with modern Python practices
- Type hints throughout codebase
- Comprehensive docstrings for all classes and methods
- `QLearningAgent` class with clean API
- Configuration management via `config.py`
- Utility functions for visualization and analysis
- Professional README with badges and documentation
- Jupyter notebook tutorial
- Interactive demo script (`demo.py`)
- Model save/load functionality
- Training progress visualization
- Q-value convergence analysis
- Multiple preset configurations
- `.gitignore` and proper project structure
- MIT License
- Contributing guidelines

### Changed
- Renamed `makeMove()` to `step()` for standard RL interface
- Renamed `calcStateNum()` to `_get_state_number()` (private method)
- Renamed `plottableBoard()` to `get_plottable_board()`
- Improved reward structure (configurable via `GameConfig`)
- Better method naming following PEP 8
- Separated concerns into multiple modules
- Enhanced visualization capabilities

### Deprecated
- Original `QLearning.py` script (use `QLearningAgent.py` instead)
- Old function names (kept for backward compatibility with warnings)

### Fixed
- Infinite loop detection in evaluation
- State space coverage analysis
- Animation generation stability
- Memory efficiency in long training runs

## [1.0.0] - Original

### Initial Implementation
- Basic Snake game environment
- Tabular Q-Learning implementation
- Simple training script
- Basic visualization
