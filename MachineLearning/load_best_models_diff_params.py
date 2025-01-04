import gym
import os
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Paths
save_dir = os.path.join("Training", "Saved_Models_Different_Parameters")
environment_name = "LunarLander-v2"

# Load environment
env = gym.make(environment_name)

# Load saved models
dqn_model_path = os.path.join(save_dir, "Best_DQN_Model.zip")
ppo_model_path = os.path.join(save_dir, "Best_PPO_Model.zip")
ddqn_model_path = os.path.join(save_dir, "Best_DDQN_Model.zip")

# Check if files exist
if not all(os.path.exists(path) for path in [dqn_model_path, ppo_model_path, ddqn_model_path]):
    raise FileNotFoundError("One or more model files are missing in the directory.")

# Load models
print("Loading models...")
dqn_model = DQN.load(dqn_model_path, env=env)
ppo_model = PPO.load(ppo_model_path, env=env)
ddqn_model = DQN.load(ddqn_model_path, env=env)  # DDQN uses the same class as DQN in stable_baselines3

# Evaluate models
def evaluate_and_display(model, env, model_name):
    print(f"Evaluating {model_name}...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(f"{model_name} - Mean Reward: {mean_reward:.2f}, Std Dev: {std_reward:.2f}")
    return mean_reward, std_reward

# Evaluate each model
evaluate_and_display(dqn_model, env, "DQN Model")
evaluate_and_display(ppo_model, env, "PPO Model")
evaluate_and_display(ddqn_model, env, "DDQN Model")

# Optional: Render one episode for each model
def render_episode(model, env, model_name):
    print(f"Rendering an episode for {model_name}...")
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
    env.close()

# Uncomment to render episodes (optional)
render_episode(dqn_model, env, "DQN Model")
render_episode(ppo_model, env, "PPO Model")
render_episode(ddqn_model, env, "DDQN Model")
