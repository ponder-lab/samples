#!/usr/bin/env python

import time
import numpy as np
import gym
import slimevolleygym

env = gym.make("SlimeVolley-v0")
env.seed(0)

obs = env.reset()

for i in range(100):
    # 000,110: do nothing
    # 100:     left
    # 010:     right
    # 001,111: jump
    # 101:     left jump
    # 011:     right jump
    action = np.random.randint(2, size=(3,))
    obs, reward, done, info = env.step(action)

    print(i)
    print("Obs:    ", obs)
    print("Reward: ", reward)
    #print("Info:   ", info)

    env.render()
    time.sleep(0.02)

    if done:
        print("done returened")
        break

env.close()
