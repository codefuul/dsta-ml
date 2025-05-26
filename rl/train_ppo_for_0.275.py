from __future__ import annotations

import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np

from til_environment import gridworld
from til_environment.flatten_dict import FlattenDictWrapper

# --- Custom Wrapper (ensure it's here) ---
class CustomSB3SeedWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.render_mode = None # Or self.venv.render_mode if it exists

    def seed(self, seed: int | None = None) -> list[int | None]:
        if seed is not None:
            # print(f"CustomSB3SeedWrapper.seed({seed}) called.")
            pass
        return [seed for _ in range(self.num_envs)]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Pass the seed and options directly to the underlying VecEnv's reset.
        # The underlying VecEnv (ProcConcatVec or ConcatVecEnv wrapped by SB3VecEnvWrapper)
        # should handle this seed according to Gymnasium API.
        obs = self.venv.reset(seed=seed, options=options)
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards, dones, infos

def main():
    # --- Environment Setup ---
    def create_pettingzoo_env():
        env = gridworld.parallel_env(
            novice=True,
            render_mode=None,
            env_wrappers=[FlattenDictWrapper]
        )
        return env

    # --- Training Parameters (same as your tuned version) ---
    total_training_steps = 750_000
    learning_rate = 7.5e-5
    n_steps_ppo = 2048
    batch_size = 64
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = 0.005
    vf_coef = 0.5
    max_grad_norm = 0.5
    num_envs_to_stack = 8 # Using 8 parallel environments
    seed = 123
    device = "cpu"

    print(f"Initializing training environment with {num_envs_to_stack} parallel environments...")

    base_pz_env_instance = create_pettingzoo_env()
    sb3_vec_env_instance = ss.pettingzoo_env_to_vec_env_v1(base_pz_env_instance)
    
    # --- RE-ENABLE MULTIPROCESSING ---
    # Use os.cpu_count() to determine number of CPUs for concat_vec_envs_v1
    # This will likely use ProcConcatVec if num_envs_to_stack > 1 and os.cpu_count() > 1
    num_cpus_for_concat = min(num_envs_to_stack, os.cpu_count() or 1)
    print(f"Using num_cpus={num_cpus_for_concat} for concat_vec_envs_v1 (multiprocessing expected if > 1).")
    
    concatenated_vec_env = ss.concat_vec_envs_v1(
        sb3_vec_env_instance,
        num_vec_envs=num_envs_to_stack,
        num_cpus=num_cpus_for_concat, # Reverted to use multiple CPUs
        base_class="stable_baselines3"
    )

    final_vec_env = CustomSB3SeedWrapper(concatenated_vec_env)
    
    # Call reset on the VecEnv WITHOUT a seed here.
    # The PPO(seed=seed) should handle the initial seeding for reproducibility.
    print("Resetting final_vec_env before PPO initialization (without explicit seed here)...")
    final_vec_env.reset() # NO EXPLICIT SEED HERE

    print(f"Starting PPO training on {device.upper()} for {total_training_steps} steps...")
    print(f"Hyperparameters: lr={learning_rate}, n_steps={n_steps_ppo}, batch_size={batch_size}, n_epochs={n_epochs}, ent_coef={ent_coef}")

    model = PPO(
        MlpPolicy,
        final_vec_env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps_ppo,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        seed=seed, # Primary seed for SB3 reproducibility
        device=device,
        tensorboard_log="./ppo_til_gridworld_tensorboard/"
    )

    start_time = time.time()
    model.learn(
        total_timesteps=total_training_steps,
        progress_bar=True
    )
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training finished in {training_duration/3600:.2f} hours.")

    # model_save_path = "ppo_til_gridworld_novice_model_tuned"
    # model.save(model_save_path)
    # print(f"Model saved to {model_save_path}.zip")
    # In train_ppo_for_0.275.py
    model_save_path = "ppo_model_reproduced_0.275_target_2.zip" 
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    print("Closing environment.")
    final_vec_env.close()

if __name__ == "__main__":
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./ppo_til_gridworld_tensorboard/", exist_ok=True)
    main()