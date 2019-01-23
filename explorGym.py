import gym
from gym import spaces
import numpy as np
# from static_env import atari_py
env = gym.make('SpaceInvaders-ram-v0')
#env = gym.make('CartPole-v0')
env.observation_space = spaces.Box(low=10, high=10, dtype=np.uint8, shape=(10,10))
print(env.observation_space)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.reset()
