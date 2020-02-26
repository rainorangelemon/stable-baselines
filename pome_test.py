import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines import POME2
from stable_baselines.pome2.policy import POMEPolicy
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)

env = make_atari_env('Pitfall-v0', num_env=4, seed=0)
env = VecFrameStack(env, n_stack=4)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = POME2(POMEPolicy, env, verbose=1)
# model.load('/Users/rainorangelemon/Downloads/pome2.zip')
model.learn(total_timesteps=100)


def evaluate(env, model, num_env, num_episode):
    """Return mean fitness (sum of episodic rewards) for given model"""
    episode_rewards = []
    done = np.array([0] * num_env)
    episode_reward = np.zeros((num_env))
    obs = env.reset()
    while num_episode >= len(episode_rewards):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        for i in range(num_env):
            if done[i]:
                episode_rewards.append(episode_reward[i])
                episode_reward[i] = 0
    return episode_rewards


def render_one_episode(env, model, num_env):
    done = np.array([0] * num_env)
    episode_reward = np.zeros((num_env))
    obs = env.reset()
    obses, next_frames = [], []
    while True:
        actions, values, next_frame, rewards_pred, states, neglogpacs = model.act_model.step(obs, None, done, need_model=True)
        obses.append(obs[0, :])
        next_frame = next_frame[0, :]
        next_frame = (next_frame * 255).astype(int)
        next_frames.append(next_frame)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(obs[0, :, :, -1])
        axes[0].axis('off')
        axes[1].imshow(next_frame)
        axes[1].axis('off')
        plt.show()

        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        for i in range(num_env):
            if done[0]:
                return obses, next_frames

obs = env.reset()
render_one_episode(env, model, 4)
env.close()
