from matplotlib import pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy


def plot_learning_curve(model_name, model, env, timesteps=500000):
    rewards = []
    for i in range(0, timesteps, 1000):
        reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        rewards.append(reward)
    plt.plot(range(0, timesteps, 1000), rewards, label=model_name)
    plt.legend()


def plot_model_comparison(models_and_rewards):
    plt.figure(figsize=(10, 5))
    for model_name, rewards in models_and_rewards.items():
        plt.plot(range(0, len(rewards) * 20000, 20000), rewards, label=model_name)
    plt.title("Learning Curve of Different Algorithms")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()


def plot_strategy_heatmap(data, title="Abundance of Strategies", save_path=None):
    plt.imshow(data, cmap="coolwarm", origin="lower")
    plt.colorbar(label="Abundance")
    plt.xlabel("Parameter 1")
    plt.ylabel("Parameter 2")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_learning_curve_multiple(models_and_rewards):
    plt.figure(figsize=(10, 5))
    for model_name, rewards in models_and_rewards.items():
        # Plot the rewards for each model
        plt.plot(range(0, len(rewards) * 1000, 1000), rewards, label=model_name)

    plt.title("Learning Curve of Different Algorithms (DQN, DDQN, PPO)")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_learning_curve_single(model_name, rewards, timesteps=500000):
    plt.figure(figsize=(8, 5))
    plt.plot(range(0, timesteps, 1000), rewards, label=model_name, color='blue')
    plt.title(f"Learning Curve of {model_name}")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
