# """Manages the RL model."""


# class RLManager:

#     def __init__(self):
#         # This is where you can initialize your model and any static
#         # configurations.
#         pass

#     def rl(self, observation: dict[str, int | list[int]]) -> int:
#         """Gets the next action for the agent, based on the observation.

#         Args:
#             observation: The observation from the environment. See
#                 `rl/README.md` for the format.

#         Returns:
#             An integer representing the action to take. See `rl/README.md` for
#             the options.
#         """

#         # Your inference code goes here.

#         return 0

"""Manages the RL model."""

import os
import numpy as np
from stable_baselines3 import PPO
from gymnasium.spaces.utils import flatten
from gymnasium.spaces import Box as GymBox 

from til_environment import gridworld
env = gridworld.env()

class RLManager:
    DUMMY_AGENT_ID = "player_0"

    def __init__(self):
        # --- MODIFICATION: Point to the model intended for 0.275 score ---
        model_filename = "ppo_model_reproduced_0.275_target_2.zip" # OR your original file name
        # model_filename = "ppo_til_gridworld_novice_model_tuned.zip" # If this was your original 0.275 model
        # --- END MODIFICATION ---
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", model_filename)
        
        try:
            self.model = PPO.load(model_path, device='cpu')
            print(f"Successfully loaded PPO model from {model_path}")
        except Exception as e:
            print(f"Error loading PPO model from {model_path}: {e}")
            raise

        try:
            temp_raw_env = gridworld.raw_env(novice=True)
            self.original_observation_space = temp_raw_env.observation_space(self.DUMMY_AGENT_ID)
            print(f"Successfully obtained original observation space structure.")
        except Exception as e:
            print(f"Error obtaining original observation space structure: {e}")
            raise

    def rl(self, observation: dict[str, int | list[int] | np.ndarray]) -> int:
        try:
            processed_observation = {}
            for key, space_def in self.original_observation_space.spaces.items():
                value = observation[key]
                if isinstance(space_def, GymBox):
                    processed_observation[key] = np.array(value, dtype=space_def.dtype)
                elif isinstance(space_def, gridworld.Discrete): 
                    processed_observation[key] = int(value)
                else:
                    processed_observation[key] = value
            flattened_obs = flatten(self.original_observation_space, processed_observation)
        except Exception as e:
            print(f"Error processing/flattening observation: {e}")
            print(f"Received observation: {observation}")
            return 4 

        action, _states = self.model.predict(flattened_obs, deterministic=True)
        return int(action)
