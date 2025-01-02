import gym
import os
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from graphs import plot_model_comparison, plot_learning_curve, plot_strategy_heatmap, \
    plot_learning_curve_multiple
from keras.models import Sequential
from keras.layers import Flatten, Dense


environment_name = "LunarLander-v2"
episodes = 5
log_dir = os.path.join("Training", "Logs")
save_dir = os.path.join("Training", "Saved_Models")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

env = gym.make(environment_name)

def test_environment(env, episodes=5):
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print(f"Episode: {episode}, Score: {score}")
    env.close()

test_environment(env)

def train_and_evaluate_model(model_class, model_name, env, total_timesteps=500000):
    print(f"Training {model_name}...")

    # Logging and saving paths
    model_log_path = os.path.join(log_dir, model_name)
    model_save_path = os.path.join(save_dir, f"{model_name}_LunarLander_model")

    model = model_class("MlpPolicy", env, verbose=1, tensorboard_log=model_log_path)

    eval_callback = EvalCallback(env, best_model_save_path=model_save_path, log_path=model_log_path, eval_freq=1000,
                                 verbose=1)

    actions_log = []

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save(model_save_path)
    print(f"{model_name} saved to {model_save_path}")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(f"{model_name} Mean Reward: {mean_reward}, Std Dev: {std_reward}")

    return model, mean_reward, actions_log

def build_ddqn_model(env):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(env.action_space.n, activation="linear"))
    return model

def train_ddqn(env, total_timesteps=500000):
    print("Training DDQN...")

    # Logging and saving paths
    model_log_path = os.path.join(log_dir, "DDQN")
    model_save_path = os.path.join(save_dir, "DDQN_LunarLander_model")

    # Initialize the model using stable-baselines3
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=model_log_path)

    # Define a callback for tracking the rewards
    eval_callback = EvalCallback(env, best_model_save_path=model_save_path, log_path=model_log_path, eval_freq=1000,
                                 verbose=1)

    # List to store actions during training
    ddqn_actions_log = []

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save the model
    model.save(model_save_path)
    print(f"DDQN saved to {model_save_path}")

    # Evaluate the model using the built-in evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(f"DDQN Mean Reward: {mean_reward}, Std Dev: {std_reward}")

    return model, ddqn_actions_log  # Return both model and actions_log



def collect_rewards(model, env, total_timesteps=500000):
    rewards = []
    for i in range(0, total_timesteps, 1000):
        reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        rewards.append(reward)
    return rewards

dqn_model, dqn_mean_reward, dqn_actions_log = train_and_evaluate_model(DQN, "DQN", env)
dqn_rewards = collect_rewards(dqn_model, env)

ddqn_model, ddqn_actions_log = train_ddqn(env)
ddqn_rewards = collect_rewards(ddqn_model, env)

ppo_model, ppo_mean_reward, ppo_actions_log = train_and_evaluate_model(PPO, "PPO", env)
ppo_rewards = collect_rewards(ppo_model, env)


plot_learning_curve('DQN', dqn_model, env)


plot_learning_curve('PPO', ppo_model, env)

plot_learning_curve('DDQN', ddqn_model, env)

plot_model_comparison({
    'DQN': dqn_rewards,
    'DDQN': ddqn_rewards,
    'PPO': ppo_rewards
})

plot_learning_curve_multiple({
    'DQN': dqn_rewards,
    'DDQN': ddqn_rewards,
    'PPO': ppo_rewards
})

q_values = np.random.rand(10, 10)
plot_strategy_heatmap(q_values)

env.close()