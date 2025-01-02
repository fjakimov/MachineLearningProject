import gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import matplotlib.pyplot as plt

save_dir = os.path.join("Training", "Saved_Models")

def load_and_render_saved_models(env, save_dir):
    saved_model_files = [f for f in os.listdir(save_dir) if f.endswith(".zip")]

    if not saved_model_files:
        print("No saved models found in the directory.")
        return

    print(f"Found saved models: {saved_model_files}")

    for model_file in saved_model_files:
        model_path = os.path.join(save_dir, model_file)
        model_name = model_file.split("_")[0]

        if model_name == "DQN" or model_name == "DDQN":
            model = DQN.load(model_path, env=env)
        elif model_name == "PPO":
            model = PPO.load(model_path, env=env)
        else:
            print(f"Unknown model type in file {model_file}, skipping...")
            continue

        print(f"Rendering gameplay for {model_name} loaded from {model_file}")

        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()

    env.close()

def plot_learning_curve(model_name, model, env, total_timesteps=500000, eval_freq=10000):

    rewards = []
    timesteps = []
    for i in range(0, total_timesteps + 1, eval_freq):
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
        rewards.append(mean_reward)
        timesteps.append(i)
        print(f"Timesteps: {i}, Mean Reward: {mean_reward}")
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, rewards, label=model_name)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title(f"Learning Curve for {model_name}")
    plt.legend()
    plt.grid()
    plt.show()

env = gym.make("LunarLander-v2")

load_and_render_saved_models(env, save_dir)


dqn_model_path = os.path.join(save_dir, "DQN_LunarLander_model.zip")
ddqn_model_path = os.path.join(save_dir, "DDQN_LunarLander_model.zip")
ppo_model_path = os.path.join(save_dir, "PPO_LunarLander_model.zip")

if os.path.exists(dqn_model_path):
    dqn_model = DQN.load(dqn_model_path, env=env)
    plot_learning_curve("DQN", dqn_model, env)

if os.path.exists(ddqn_model_path):
    ddqn_model = DQN.load(ddqn_model_path, env=env)
    plot_learning_curve("DDQN", ddqn_model, env)

if os.path.exists(ppo_model_path):
    ppo_model = PPO.load(ppo_model_path, env=env)
    plot_learning_curve("PPO", ppo_model, env)
