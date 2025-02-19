#!/usr/bin/env python3
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime
import argparse
from environment.door_env import Door

FINAL_MODEL_PATH = ""
INTERMEDIATE_MODEL_PATH = ""
TOTAL_TIMESTEPS = 0
REMAINING_TIMESTEPS = 0

# -------------------------------------------------------------------------------
# NOTE [CS5446]: this is the part where we should decide what RL algorithm to use
# I'm using PPO from stable_baselines3 as an example here
# -------------------------------------------------------------------------------
def train(env, resume=False):
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./models/",
        name_prefix="ppo_panda_door",
        save_replay_buffer=True,
    )
    if resume:
        model = PPO.load(f"{INTERMEDIATE_MODEL_PATH}", env)
        model.learn(
            total_timesteps=REMAINING_TIMESTEPS,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
        )
    else:
        # Define the model
        model = PPO("MlpPolicy", env, verbose=1)

        # Train the model
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"{FINAL_MODEL_PATH}_{current_time}_{str(TOTAL_TIMESTEPS)}")
    env.close()


def test(env):
    model = PPO.load(f"{FINAL_MODEL_PATH}")

    print("Model Architecture:")
    print(model.policy)
    print("\nModel Hyperparameters:")
    print(f"learning_rate: {model.learning_rate}")  # Example of a hyperparameter
    print(f"n_steps: {model.n_steps}")  # Number of steps per update
    print(
        f"batch_size: {model.batch_size}"
    )  # Size of the batch for each training update
    print(f"gamma: {model.gamma}")  # Discount factor for rewards
    print(
        f"gae_lambda: {model.gae_lambda}"
    )  # GAE (Generalized Advantage Estimation) lambda

    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, _, info = env.step(action)
        env.render()
        if dones:
            print(f"Task completed! Total rewards: {rewards}")
            break
    env.close()
    print("Finish testing...")


def main(args):
    if args.use_default_env:
        env = suite.make(
            env_name="Door",
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
        )
    else:
        env = Door(
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
        )
    env = GymWrapper(env)

    if args.mode.lower() == "train":
        if args.resume:
            print("Resume training the model...")
            train(env, True)
        else:
            print("Training the model...")
            train(env)
    else:
        print("Testing the model...")
        test(env)
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Train the model")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training the model"
    )
    parser.add_argument(
        "--use_default_env", action="store_true", help="Use default environment"
    )
    parser.add_argument(
        "--final_model_path",
        type=str,
        default="ppo_panda_door_final",
        help="Final model file path",
    )
    parser.add_argument(
        "--intermediate_model_path",
        type=str,
        default="ppo_panda_door_final",
        help="Intermediate model file path",
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=6000000, help="Total number of timesteps"
    )
    parser.add_argument(
        "--remaining_timesteps",
        type=int,
        default=0,
        help="Remaining number of timesteps",
    )
    args = parser.parse_args()

    FINAL_MODEL_PATH = args.final_model_path
    INTERMEDIATE_MODEL_PATH = args.intermediate_model_path
    TOTAL_TIMESTEPS = args.total_timesteps
    REMAINING_TIMESTEPS = args.remaining_timesteps
    main(args)
