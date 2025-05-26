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
        self.render_mode = None

    def seed(self, seed: int | None = None) -> list[int | None]:
        return [seed for _ in range(self.num_envs)]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return self.venv.reset(seed=seed, options=options)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards, dones, infos

def main():
    def create_pettingzoo_env():
        env = gridworld.parallel_env(
            novice=True,
            render_mode=None,
            env_wrappers=[FlattenDictWrapper]
        )
        return env

    # --- Training Parameters (optimized for value loss and explained variance) ---
    total_training_steps = 250_000  # Increased steps for better convergence
    learning_rate = 3e-4              # Lower LR for smoother updates
    n_steps_ppo = 1024                # Larger batch to stabilize value function
    batch_size = 256                 # Match with n_steps
    n_epochs = 15                    # More epochs to improve critic
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.1                 # Tighter clip range to stabilize updates
    ent_coef = 0.0                # Less entropy to focus on value optimization
    vf_coef = 0.8  
    target_kl = 0.015
    max_grad_norm = 0.2
    num_envs_to_stack = 8
    normalize_advantage=True
    seed = 86
    device = "cpu"

    print(f"Initializing training environment with {num_envs_to_stack} parallel environments...")

    base_pz_env_instance = create_pettingzoo_env()
    sb3_vec_env_instance = ss.pettingzoo_env_to_vec_env_v1(base_pz_env_instance)

    num_cpus_for_concat = min(num_envs_to_stack, os.cpu_count() or 1)
    print(f"Using num_cpus={num_cpus_for_concat} for concat_vec_envs_v1.")

    concatenated_vec_env = ss.concat_vec_envs_v1(
        sb3_vec_env_instance,
        num_vec_envs=num_envs_to_stack,
        num_cpus=num_cpus_for_concat,
        base_class="stable_baselines3"
    )

    final_vec_env = CustomSB3SeedWrapper(concatenated_vec_env)
    final_vec_env.reset()

    print(f"Starting PPO training on {device.upper()} for {total_training_steps} steps...")

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
        target_kl=target_kl,
        seed=seed,
        device=device,
        tensorboard_log="./ppo_til_gridworld_tensorboard/"
    )

    start_time = time.time()
    model.learn(
        total_timesteps=total_training_steps,
        progress_bar=True
    )
    end_time = time.time()
    print(f"Training finished in {(end_time - start_time) / 3600:.2f} hours.")

    model_save_path = "ppo_til_gridworld_novice_model_tuned"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}.zip")

    final_vec_env.close()

if __name__ == "__main__":
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./ppo_til_gridworld_tensorboard/", exist_ok=True)
    main()
