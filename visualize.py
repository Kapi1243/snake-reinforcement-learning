"""
Create GIF/images of trained Snake AI for website.
Usage: python visualize.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from pathlib import Path
from src.Snake import SnakeGame
from src.QLearningAgent import QLearningAgent

# Custom colormap: white (empty), black (snake), red (food)
COLORS = ListedColormap(["#FFFFFF", "#000000", "#FF0000"])


def create_game_gif(agent, board_size=12, num_games=3, save_path="Images/snake_gameplay.gif"):
    """Create animated GIF of the snake playing."""

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")

    all_frames = []

    print(f"Recording {num_games} games...")

    for game_num in range(num_games):
        game = SnakeGame(width=board_size, height=board_size)
        state = game.reset()
        done = False

        while not done:
            # Get current board state
            board_copy = game.board.copy()
            all_frames.append(board_copy)

            # Play move
            action = agent.select_action(state, training=False)
            state, reward, done, score = game.step(action)

        print(f"  Game {game_num + 1}: Score {score}")

    # Create animation
    print(f"\nCreating GIF with {len(all_frames)} frames...")

    im = ax.imshow(all_frames[0], cmap=COLORS, vmin=0, vmax=2, animated=True)

    def update_frame(frame_num):
        im.set_data(all_frames[frame_num])
        return [im]

    ani = animation.FuncAnimation(
        fig, update_frame, frames=len(all_frames), interval=100, blit=True, repeat=True
    )

    # Save
    Path(save_path).parent.mkdir(exist_ok=True)
    ani.save(save_path, writer="pillow", fps=10)
    print(f"GIF saved to: {save_path}")
    plt.close()


def create_static_image(agent, board_size=12, save_path="Images/snake_snapshot.png"):
    """Create a static image of the snake playing."""

    game = SnakeGame(width=board_size, height=board_size)
    state = game.reset()

    # Play a few moves to get interesting state
    for _ in range(20):
        action = agent.select_action(state, training=False)
        state, reward, done, score = game.step(action)
        if done:
            break

    # Render
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(game.board, cmap=COLORS, vmin=0, vmax=2)
    ax.axis("off")
    ax.set_title(f"Snake AI - Score: {score}", fontsize=16, pad=20, color="black")
    fig.patch.set_facecolor("#ffffff")

    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Image saved to: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("SNAKE AI VISUALIZATION")
    print("=" * 60)

    # Train a quick agent
    print("\nTraining AI agent...")
    agent = QLearningAgent(gamma=0.9, learning_rate=0.3, epsilon=0.3)
    agent.train(board_size=12, num_episodes=2000, verbose=False)

    print("\n" + "=" * 60)
    print("Creating visuals...")
    print("=" * 60)

    # Create GIF
    create_game_gif(agent, board_size=12, num_games=3, save_path="Images/snake_gameplay.gif")

    # Create static image
    create_static_image(agent, board_size=12, save_path="Images/snake_snapshot.png")

    print("\n" + "=" * 60)
    print("Done! Check the Images/ folder")
    print("=" * 60)


if __name__ == "__main__":
    main()
