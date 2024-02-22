"""
Script for training the model for atari games.
"""
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

LR = 0.001
BUFFER_SIZE = 10_000
LEARNING_STARTS = 100
BATCH_SIZE = 100
TAU = 0.005
GAMMA = 0.99
SEED = 123

TOTAL_TIMESTEPS = 10_000
LOG_INTERVAL = 10


def train():
    # Create environment
    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    # Instantiate the agent
    model = DQN("MlpPolicy",
                env,
                verbose=1,
                learning_rate=LR,
                batch_size=BATCH_SIZE,
                learning_starts=LEARNING_STARTS,
                tau=TAU,
                gamma=GAMMA,

                )

    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e5), progress_bar=True, log_interval=LOG_INTERVAL)
    # Save the agent
    model.save("dqn_lunar")
    return model, env


def inference(env: gym.Env = None, model: DQN = None):

    if env is None:
        env = gym.make("LunarLander-v2", render_mode="rgb_array")

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    if model is None:
        model = DQN.load("dqn_lunar", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    inference()
