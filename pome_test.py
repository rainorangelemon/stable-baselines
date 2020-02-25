import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines import POME2
from stable_baselines.pome2.policy import POMEPolicy
import numpy as np
from tqdm import tqdm
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

env = make_atari_env('RoadRunner-v0', num_env=4, seed=0)
env = VecFrameStack(env, n_stack=4)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = POME2(POMEPolicy, env, verbose=1)
model.learn(total_timesteps=10000)


def evaluate(env, model, num_env, iter_step):
    """Return mean fitness (sum of episodic rewards) for given model"""
    episode_rewards = []
    episode_reward = np.zeros((num_env))
    obs = env.reset()
    for _ in tqdm(range(iter_step)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        for i in range(num_env):
            if done[i]:
                episode_rewards.append(episode_reward[i])
                episode_reward[i] = 0
    return episode_rewards


obs = env.reset()
rewards = evaluate(env, model, 4, 10000)
print(rewards)
env.close()
