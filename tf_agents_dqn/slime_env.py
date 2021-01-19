#!/usr/bin/env python

import numpy as np
import gym
import slimevolleygym

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils

def convert_action(action):
    if action == 0:
        return np.array([0, 0, 0])
    elif action == 1:
        return np.array([1, 0, 0])
    elif action == 2:
        return np.array([0, 1, 0])
    elif action == 3:
        return np.array([0, 0, 1])
    elif action == 4:
        return np.array([1, 0, 1])
    elif action == 5:
        return np.array([0, 1, 1])
    else:
        assert False, "Invalid aciton"

class slime_env(py_environment.PyEnvironment):

    def __init__(self):
        self.env = gym.make("SlimeVolley-v0")

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=5, name="action")

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(12,), dtype=np.float64,
            minimum=np.finfo(np.float32).min,
            maximum=np.finfo(np.float32).max, name="observation")

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
        self.env.viewer = None

    def _reset(self):
        self.prev_bx = 0.0
        self.prev_bdy = 0.0
        return ts.restart(self.env.reset())
    
    def _step(self, action):

        action_bin = convert_action(action)
        
        obs = self.env.step(action_bin)

        #reward = obs[1]

        #reward = obs[3]["ale.lives"] - obs[3]["ale.otherLives"]

        #if obs[0][4] > 0.0 and self.prev_bdy < 0.0 and obs[0][7] > 0.0:
        #    reward = 0.5
        #else:
        #    reward = obs[1]

        if obs[0][4] < 0.0 and self.prev_bx > 0.0:
            reward = 0.5
        elif obs[0][4] > 0.0 and self.prev_bdy < 0.0 and obs[0][7] > 0.0 and obs[0][6] < 0.0:
            reward = 0.1
        else:
            reward = obs[1]
        self.prev_bx = obs[0][4]
        self.prev_bdy = obs[0][7]

        if obs[2]: # if done
            return ts.termination(obs[0], reward)
        else:
            return ts.transition(obs[0], reward=reward, discount=0.9)

def main():
    env = slime_env()
    #o0 = env.reset()
    #o1 = env.step(0)

    # #Check if the class work ok
    utils.validate_py_environment(env, episodes=5)

if __name__ == "__main__":
    main()
