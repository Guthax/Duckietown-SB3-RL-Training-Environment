import gym
from gym import spaces
import numpy as np

class CustomGymEnv(gym.Env):
    """
    Custom OpenAI Gym environment with a composite observation space
    including an RGB camera, VAE latent space, and vehicle metrics.
    """
    def __init__(self):
        # Define observation space
        self.observation_space = spaces.Dict({
            "rgb_camera": gym.spaces.Box(low=0, high=255, shape=(80, 160, 3), dtype=np.uint8),
            "vae_latent": spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32),
            "vehicle_measures": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })

        # Define action space (example: discrete actions for simplicity)
        self.action_space  = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)

    def reset(self):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        observation = {
            "rgb_camera": np.zeros((80, 120, 3), dtype=np.uint8),
            "vae_latent": np.random.normal(size=(64,)).astype(np.float32),
            "vehicle_metrics": np.zeros((3,), dtype=np.float32)
        }
        return observation

    def step(self, action):
        """
        Executes a step in the environment using the given action.
        Returns a tuple (observation, reward, done, info).
        """
        # Example observation update
        observation = {
            "rgb_camera": np.random.randint(0, 256, size=(80, 120, 3), dtype=np.uint8),
            "vae_latent": np.random.normal(size=(64,)).astype(np.float32),
            "vehicle_metrics": np.random.random(size=(3,)).astype(np.float32)
        }

        # Example reward and done flag
        reward = 0.0
        done = False

        # Example info (can include diagnostic data)
        info = {}

        return observation, reward, done, info

    def render(self, mode='human'):
        """
        Renders the environment (example: show RGB camera feed).
        """
        pass  # Replace with rendering logic if needed

    def close(self):
        """
        Cleans up resources when the environment is closed.
        """
        pass
