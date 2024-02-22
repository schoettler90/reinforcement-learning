import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

# Hyperparameters
LR = 0.001
BUFFER_SIZE = 10_000
LEARNING_STARTS = 100
BATCH_SIZE = 100
TAU = 0.005
GAMMA = 0.99
SEED = 123

TOTAL_TIMESTEPS = 10_000
LOG_INTERVAL = 10


def train() -> (DDPG, make_vec_env):
    env = make_vec_env("Pendulum-v1", n_envs=1, seed=SEED)

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy",
                 env,
                 learning_rate=LR,
                 buffer_size=BUFFER_SIZE,
                 learning_starts=LEARNING_STARTS,
                 batch_size=BATCH_SIZE,
                 tau=TAU,
                 gamma=GAMMA,
                 verbose=1,
                 action_noise=action_noise,
                 )

    print("Model:")
    print(model)
    print(model.policy)

    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                log_interval=LOG_INTERVAL)
    model.save("ddpg_pendulum_noise")
    vec_env = model.get_env()

    print("Finished training.")

    return model, vec_env


if __name__ == "__main__":
    train()
