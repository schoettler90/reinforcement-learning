from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("Pendulum-v1", n_envs=1, seed=123)
model = DDPG.load("ddpg_pendulum_noise")
# vec_env = model.get_env()

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render(mode="human")
