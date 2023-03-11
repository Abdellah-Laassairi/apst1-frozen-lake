import sys
sys.path.append(r"C:/ECN_2022/APST2/")
import logging
import itertools
import numpy as np
np.random.seed(0)
import gymnasium as gym
import matplotlib.pyplot as plt
from A2C_torch import *

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s [%(levelname)s] %(message)s', 
                    stream=sys.stdout, 
                    datefmt='%H:%M:%S')

## create agent
env = gym.make("Acrobot-v1")
np.random.seed(42)
observation, _ = env.reset()
agent = A2C(env)

## train
logging.info('==== train ====')
episode_rewards = []
for episode in itertools.count():
    episode_reward, elapsed_steps = play_epis(env, agent, 
                                         max_episode_steps=env._max_episode_steps, 
                                         mode='train', render=1)
    episode_rewards.append(episode_reward)
    logging.debug('train episode %d: reward = %.2f, steps = %d', 
                  episode, episode_reward, elapsed_steps)
    if np.mean(episode_rewards[-10:])>-120:
        break
plt.plot(episode_rewards)

## test
logging.info('==== test ====')
episode_rewards = []
for episode in range(100):
    episode_reward, elapsed_steps = play_epis(env, agent)
    episode_rewards.append(episode_reward)
    logging.debug('test episode %d: reward = %.2f, steps = %d', 
                  episode, episode_reward, elapsed_steps)
logging.info('average episode reward = %.2f', 
             np.mean(episode_rewards))

env.close()
