#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym

# # Random control
# frames = []
# env = gym.make("CartPole-v0")
# observation = env.reset()
# 
# fig = plt.figure()
# 
# for step in range(0, 200):
#     img = env.render(mode="rgb_array")
#     frames += [[plt.imshow(img)]]
#     action = np.random.choice(2)
# 
#     observation, reward, done, info = env.step(action)
# 
# anim = animation.ArtistAnimation(fig, frames, interval=100)
# anim.save("anim.gif", writer="pillow", fps=30)

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation, state_vrange):
    pos, v, angle, ang_v = observation

    d_pos   = np.digitize(pos,   bins=bins(-2.4, 2.4, state_vrange))
    d_v     = np.digitize(v,     bins=bins(-3.0, 3.0, state_vrange))
    d_angle = np.digitize(angle, bins=bins(-0.5, 0.5, state_vrange))
    d_ang_v = np.digitize(ang_v, bins=bins(-2.0, 2.0, state_vrange))

    state = d_pos + d_v * 6 + d_angle * 6**2 + d_ang_v * 6**3
    
    return state

def decide_action(q_table, state, episode, action_vrange):
    epsilon = 0.5 * (1.0 / (episode + 1))

    if epsilon <= np.random.uniform(0, 1):
        action = np.argmax(q_table[state])
    else:
        action = np.random.choice(action_vrange)

    return action

def update_Q_function(q_table, state, action, reward, state_next, eta, gamma):
    Max_Q_next = max(q_table[state_next])
    q_table[state, action] = q_table[state, action] + \
                             eta * (reward + gamma * Max_Q_next - q_table[state, action])
    

np.random.seed(0)

num_episodes = 1000
max_steps = 200
num_states = 4
state_vrange = 6
action_vrange = 2
eta = 0.5
gamma = 0.99

env = gym.make("CartPole-v0")
q_table = np.random.uniform(0, 1, size=(state_vrange**num_states, action_vrange))

num_completes = 0
final_episode = False
frames = []
fig = plt.figure()

for episode in range(num_episodes):

    observation = env.reset()

    for step in range(max_steps):

        if final_episode:
            img = env.render(mode="rgb_array")
            frames += [[plt.imshow(img)]]

        state = digitize_state(observation, state_vrange)
        action = decide_action(q_table, state, episode, action_vrange)

        observation_next, _, done, _ = env.step(action)

        if done:
            if step < 195:
                reward = -1
                num_completes = 0
            else:
                reward = 1
                num_completes += 1
        else:
            reward = 0

        state_next = digitize_state(observation_next, state_vrange)
        update_Q_function(q_table, state, action, reward, state_next, eta, gamma)

        observation = observation_next

        if done:
            print("Episode {}: finished after {} time steps".format(episode, step + 1))
            break

    if final_episode:
        anim = animation.ArtistAnimation(fig, frames, interval=100)
        anim.save("anim.gif", writer="pillow", fps=30)
        break

    if num_completes >= 10:
        print("10 consecutive successes")
        final_episode = True

