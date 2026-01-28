"""
Quick demo script showing different use cases of the Snake RL project.

Run this script to see the agent in action!
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from Snake import SnakeGame, play_human_game
from QLearningAgent import QLearningAgent
from config import PresetConfigs
from utils import plot_training_curves, analyze_state_coverage, print_training_summary


def demo_human_play():
    """Demo: Play Snake yourself."""
    print("\n" + "=" * 60)
    print("DEMO 1: Human Player")
    print("=" * 60)
    print("\nYou can play the game yourself!")
    print("Controls: W=Up, A=Left, S=Down, D=Right, Q=Quit\n")

    response = input("Would you like to play? (y/n): ")
    if response.lower() == "y":
        play_human_game(board_size=8)


def demo_quick_training():
    """Demo: Quick training session."""
    print("\n" + "=" * 60)
    print("DEMO 2: Quick Training (1000 episodes)")
    print("=" * 60)
    print("\nThis will train an agent for 1000 episodes on an 8x8 board.")
    print("Should take less than a minute.\n")

    response = input("Start training? (y/n): ")
    if response.lower() != "y":
        return

    # Get quick test configuration
    game_config, training_config = PresetConfigs.quick_test()

    # Create and train agent
    agent = QLearningAgent(
        gamma=training_config.gamma,
        epsilon=training_config.epsilon,
        learning_rate=training_config.learning_rate,
    )

    results = agent.train(
        board_size=game_config.board_width,
        num_episodes=training_config.num_episodes,
        eval_frequency=training_config.eval_frequency,
        eval_episodes=training_config.eval_episodes,
        save_q_history=False,
        verbose=True,
    )

    # Analyze results
    coverage = analyze_state_coverage(agent.q_table)
    print_training_summary(results, coverage)

    # Show plot
    agent.plot_training_progress()

    return agent


def demo_evaluation(agent=None):
    """Demo: Evaluate a trained agent."""
    print("\n" + "=" * 60)
    print("DEMO 3: Agent Evaluation")
    print("=" * 60)

    if agent is None:
        print("\nNo agent provided. Training a quick one...")
        game_config, training_config = PresetConfigs.quick_test()
        agent = QLearningAgent(gamma=training_config.gamma, epsilon=training_config.epsilon)
        agent.train(board_size=game_config.board_width, num_episodes=1000, verbose=False)

    print("\nEvaluating agent performance over 50 games...")
    avg_score, scores = agent.evaluate(board_size=8, num_episodes=50)

    print(f"\nResults:")
    print(f"  Average Score: {avg_score:.2f}")
    print(f"  Best Game: {max(scores)}")
    print(f"  Worst Game: {min(scores)}")
    print(f"  Games with score ≥10: {sum(1 for s in scores if s >= 10)}/50")


def demo_watch_agent():
    """Demo: Watch the agent play (text-based)."""
    print("\n" + "=" * 60)
    print("DEMO 4: Watch Agent Play")
    print("=" * 60)
    print("\nTraining a quick agent and watching it play...")

    # Train agent
    agent = QLearningAgent(epsilon=0.2, gamma=0.8)
    agent.train(board_size=8, num_episodes=1000, verbose=False)

    print("\n" + "=" * 60)
    print("Agent is now playing! Watch the O (head) and X (body)")
    print("=" * 60 + "\n")

    # Play one game with visualization
    game = SnakeGame(8, 8)
    state = game._get_state_number()
    game_over = False
    steps = 0
    max_steps = 100

    game.display()
    print(f"Score: {game.length}\n")

    input("Press Enter to start...")

    while not game_over and steps < max_steps:
        action = agent.select_action(state, training=False)
        state, reward, game_over, score = game.step(action)

        direction_names = ["UP", "RIGHT", "DOWN", "LEFT"]
        print(f"Action: {direction_names[action]} | Reward: {reward:+.0f}")
        game.display()
        print(f"Score: {score}\n")

        if not game_over:
            import time

            time.sleep(0.3)  # Slow down for visibility

        steps += 1

    if game_over:
        print("Game Over!")
    else:
        print("Maximum steps reached.")

    print(f"Final Score: {score}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("🐍 SNAKE Q-LEARNING DEMO SCRIPT 🐍")
    print("=" * 60)
    print("\nThis script demonstrates various features of the project.")

    while True:
        print("\n" + "=" * 60)
        print("Available Demos:")
        print("=" * 60)
        print("1. Play Snake yourself (human)")
        print("2. Quick training demonstration")
        print("3. Agent evaluation")
        print("4. Watch agent play (animated text)")
        print("5. Run all demos")
        print("0. Exit")

        choice = input("\nSelect a demo (0-5): ").strip()

        if choice == "0":
            print("\nThanks for trying the demo! 🐍")
            break
        elif choice == "1":
            demo_human_play()
        elif choice == "2":
            agent = demo_quick_training()
            if agent and input("\nEvaluate this agent? (y/n): ").lower() == "y":
                demo_evaluation(agent)
        elif choice == "3":
            demo_evaluation()
        elif choice == "4":
            demo_watch_agent()
        elif choice == "5":
            demo_human_play()
            agent = demo_quick_training()
            demo_evaluation(agent)
            demo_watch_agent()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
