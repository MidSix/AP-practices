"""Entorno de conducción con imágenes RGB de 48 × 48 y acciones discretas."""
import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.registration import register, registry

ENV_ID = "AP-CarRacing48-v1"


class CarRacing48(CarRacing):
    metadata = {"render_modes": ["rgb_array", "state_pixels"], "render_fps": 12}

    def __init__(self, render_mode=None):
        if render_mode not in (None, "rgb_array", "state_pixels"):
            raise ValueError("Solo se admite None, 'rgb_array' o 'state_pixels'.")
        super().__init__(render_mode=render_mode, continuous=False,
                         domain_randomize=False)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(48, 48, 3), dtype=np.uint8)

    def _create_image_array(self, screen, size):


        return super()._create_image_array(screen, (48, 48))

    def step(self, action):

        if action is None:
            return super().step(None)
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not self.action_space.contains(action):
            raise ValueError("Acción entera: 0=nada, 1=derecha, 2=izquierda, 3=acelerar, 4=frenar.")
        action = int(action)
        total_reward = 0.0
        for _ in range(4):
            obs, reward, terminated, truncated, info = super().step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

    def render(self):

        if getattr(self, "state", None) is None:
            raise gym.error.ResetNeeded("Llama a reset() antes de render().")
        return self.state.copy()


if ENV_ID not in registry:
    register(id=ENV_ID, entry_point=CarRacing48, max_episode_steps=250)


def make_env(seed=None, *, max_episode_steps=250):
    """Devuelve un entorno sin resetear; seed inicializa solo las acciones."""
    env = gym.make(ENV_ID, render_mode="rgb_array", max_episode_steps=max_episode_steps)
    if seed is not None:
        env.action_space.seed(seed)
    return env
