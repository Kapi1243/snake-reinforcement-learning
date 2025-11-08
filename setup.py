"""
Setup configuration for Snake Reinforcement Learning package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="snake-rl",
    version="2.0.0",
    author="Kacper Kowalski",
    description="A modern Q-Learning implementation for the Snake game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kapi1243/snake-reinforcement-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "snake-rl-demo=src.demo:main",
            "snake-rl-play=src.Snake:play_human_game",
        ],
    },
    keywords="reinforcement-learning q-learning snake machine-learning ai",
    project_urls={
        "Bug Reports": "https://github.com/Kapi1243/snake-reinforcement-learning/issues",
        "Source": "https://github.com/Kapi1243/snake-reinforcement-learning",
    },
)
