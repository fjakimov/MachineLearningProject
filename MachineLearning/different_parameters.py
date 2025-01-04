import gym
import os
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


environment_name = "LunarLander-v2"
log_dir = os.path.join("Training", "Logs")
save_dir = os.path.join("Training", "Saved_Models_Different_Parameters")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

env = gym.make(environment_name)

def train_and_evaluate_model_with_hyperparams(model_class, model_name, env, hyperparams, total_timesteps=500000):
    print(f"Training {model_name} with hyperparameters: {hyperparams}...")

    # Logging and saving paths
    model_log_path = os.path.join(log_dir, model_name)
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    model = model_class("MlpPolicy", env, verbose=1, tensorboard_log=model_log_path, **hyperparams)

    eval_callback = EvalCallback(env, best_model_save_path=model_save_dir, log_path=model_log_path, eval_freq=1000,
                                 verbose=1)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save(os.path.join(model_save_dir, f"{model_name}_final"))
    print(f"{model_name} saved to {model_save_dir}")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(f"{model_name} Mean Reward: {mean_reward}, Std Dev: {std_reward}")

    return model, mean_reward


dqn_hyperparams = [
    {"learning_rate": 1e-3, "buffer_size": 100000, "batch_size": 32, "gamma": 0.99},
    {"learning_rate": 5e-4, "buffer_size": 50000, "batch_size": 64, "gamma": 0.98},
    {"learning_rate": 1e-4, "buffer_size": 200000, "batch_size": 128, "gamma": 0.95}
]

ppo_hyperparams = [
    {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "gae_lambda": 0.95},
    {"learning_rate": 1e-4, "n_steps": 1024, "batch_size": 128, "gae_lambda": 0.99},
    {"learning_rate": 5e-4, "n_steps": 4096, "batch_size": 32, "gae_lambda": 0.90}
]
ddqn_hyperparams = [
    {"learning_rate": 1e-3, "buffer_size": 100000, "batch_size": 64, "gamma": 0.99, "target_update_interval": 1000},
    {"learning_rate": 5e-4, "buffer_size": 200000, "batch_size": 128, "gamma": 0.95, "target_update_interval": 500},
    {"learning_rate": 1e-4, "buffer_size": 50000, "batch_size": 32, "gamma": 0.98, "target_update_interval": 2000}
]

def train_ddqn_with_hyperparams(env, hyperparams, total_timesteps=500000):
    print(f"Training DDQN with hyperparameters: {hyperparams}...")

    model_name = "DDQN"
    model_log_path = os.path.join(log_dir, model_name)
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=model_log_path, **hyperparams)

    eval_callback = EvalCallback(env, best_model_save_path=model_save_dir, log_path=model_log_path, eval_freq=1000,
                                 verbose=1)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(model_save_dir, f"{model_name}_final"))
    print(f"DDQN model saved to {model_save_dir}")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(f"DDQN Mean Reward: {mean_reward}, Std Dev: {std_reward}")

    return model, mean_reward

ddqn_results = []
for idx, params in enumerate(ddqn_hyperparams):
    model, mean_reward = train_ddqn_with_hyperparams(env, params)
    ddqn_results.append((f"DDQN_Params_{idx+1}", model, mean_reward))


dqn_results = []
for idx, params in enumerate(dqn_hyperparams):
    model_name = f"DQN_Params_{idx+1}"
    model, mean_reward = train_and_evaluate_model_with_hyperparams(DQN, model_name, env, params)
    dqn_results.append((model_name, model, mean_reward))

ppo_results = []
for idx, params in enumerate(ppo_hyperparams):
    model_name = f"PPO_Params_{idx+1}"
    model, mean_reward = train_and_evaluate_model_with_hyperparams(PPO, model_name, env, params)
    ppo_results.append((model_name, model, mean_reward))

def collect_rewards(model, env, total_timesteps=50000):
    rewards = []
    for _ in range(10):  # Evaluate over 10 episodes
        reward, _ = evaluate_policy(model, env, n_eval_episodes=1)
        rewards.append(reward)
    return rewards

best_dqn_model = max(dqn_results, key=lambda x: x[2])[1]
best_ppo_model = max(ppo_results, key=lambda x: x[2])[1]
best_ddqn_model = max(ddqn_results, key=lambda x: x[2])[1]
best_ddqn_model.save(os.path.join(save_dir, "Best_DDQN_Model"))
# Save the best models with specific names
best_dqn_model.save(os.path.join(save_dir, "Best_DQN_Model"))
best_ppo_model.save(os.path.join(save_dir, "Best_PPO_Model"))

print("Best DQN,PPO,DDQN models saved.")
def plot_rewards(results, title):
    plt.figure(figsize=(10, 6))
    for model_name, model, mean_reward in results:
        rewards = collect_rewards(model, env)  # Collect rewards for plotting
        plt.plot(rewards, label=f"{model_name} (Mean Reward: {mean_reward:.2f})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

print("Plotting DQN Rewards...")
plot_rewards(dqn_results, "DQN Models with Different Hyperparameters")

print("Plotting PPO Rewards...")
plot_rewards(ppo_results, "PPO Models with Different Hyperparameters")

print("Plotting DDQN Rewards...")
plot_rewards(ddqn_results, "DDQN Models with Different Hyperparameters")
env.close()
