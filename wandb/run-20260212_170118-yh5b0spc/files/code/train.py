import argparse
import os
import gymnasium as gym
import envs  # Register environment
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback

def train(args):
    # Create log dir
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = "./models/"
    os.makedirs(model_dir, exist_ok=True)

    run = wandb.init(
        project="RC-Humanoid",
        config=vars(args),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    env = make_vec_env(
        "BoosterT1-v0",
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render_mode": "rgb_array"},
    )

    # Define model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=model_dir, name_prefix="booster_ppo"
    )

    # Train
    print(f"Starting training for {args.timesteps} timesteps...")
    
    callbacks = [
        checkpoint_callback,
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    
    # Save final model
    model.save(f"{model_dir}/booster_ppo_final")
    print("Training finished and model saved.")
    run.finish()

def test(args):
    model_path = args.model_path
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}")
        return

    env = gym.make("BoosterT1-v0", render_mode="human")
    model = PPO.load(model_path)

    obs, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    print("Starting test run...")
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        env.render()

    print(f"Test finished. Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Training timesteps")
    parser.add_argument("--n_envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--model_path", type=str, default="./models/booster_ppo_final", help="Path to model for testing")
    
    args = parser.parse_args()

    if args.test:
        test(args)
    else:
        train(args)
