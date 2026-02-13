import gymnasium as gym
import envs # This registers the env

env = gym.make("BoosterT1-v0", render_mode="rgb_array")
print("Env created")

obs, info = env.reset()
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Initial obs shape: {obs.shape}")

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Step done. Reward: {reward}, Terminated: {terminated}")

env.close()
