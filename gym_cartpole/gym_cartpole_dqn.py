#!/usr/bin/env python

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym

import numpy as np
import tensorflow as tf

class replay_memory:
    def __init__(self, capacity, num_states):
        self.capacity = capacity
        self.state = np.empty((0, num_states), np.float32)
        self.action = np.empty((0,), np.int64)
        self.state_n = np.empty((0, num_states), np.float32)
        self.reward = np.empty((0,), np.float32)

    def push(self, state, action, state_n, reward):
        if len(self.state) < self.capacity:
            self.state = np.append(self.state, np.array([state], dtype=np.float32), axis=0)
            self.action = np.append(self.action, np.array([action], dtype=np.int64), axis=0)
            self.state_n = np.append(self.state_n, np.array([state_n], dtype=np.float32), axis=0)
            self.reward = np.append(self.reward, np.array([reward], dtype=np.float32), axis=0)
        else:
            self.state = np.roll(self.state, -1, axis=0)
            self.action = np.roll(self.action, -1, axis=0)
            self.state_n = np.roll(self.state_n, -1, axis=0)
            self.reward = np.roll(self.reward, -1, axis=0)
            self.state[-1] = state
            self.action[-1] = action
            self.state_n[-1] = state_n
            self.reward[-1] = reward

    def sample(self, batch_size):
        rand_idx = np.random.randint(self.state.shape[0], size=batch_size)
        state_batch = self.state[rand_idx, :]
        action_batch = self.action[rand_idx]
        state_n_batch = self.state_n[rand_idx, :]
        reward_batch = self.reward[rand_idx]

        return state_batch, action_batch, state_n_batch, reward_batch

    def __len__(self):
        return len(self.state)

def generate_model(num_states, num_actions):
    input_node = tf.keras.Input((num_states,))
    x = input_node
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(num_actions)(x)
    output_node = x
    return tf.keras.models.Model(input_node, output_node)

def replay(memory, model, batch_size, optimizer, gamma):
    if len(memory) < batch_size:
        return

    state_batch, action_batch, state_n_batch, reward_batch = memory.sample(batch_size)

    select_idx = np.hstack([np.arange(batch_size).reshape(batch_size, 1),
                            action_batch.reshape(batch_size, 1)])
    
    next_q_values = model(state_n_batch)
    next_state_action_values = tf.math.reduce_max(next_q_values, axis=1)
    expected_state_action_values = reward_batch + gamma * next_state_action_values

    with tf.GradientTape() as tape:
        q_values = model(state_batch)
        state_action_values = tf.gather_nd(q_values, select_idx)
        loss = tf.keras.losses.Huber()(expected_state_action_values, state_action_values)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def decide_action(model, state, episode, action_vrange):
    epsilon = 0.5 * (1.0 / (episode + 1))

    if epsilon <= np.random.uniform(0, 1):
        s = np.array(state, dtype=np.float32).reshape(1, -1)
        q_value = model(s)
        action = np.argmax(q_value.numpy())
    else:
        action = np.random.choice(action_vrange)

    return action


random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

num_episodes = 500
max_steps = 200
gamma = 0.99
batch_size = 32
capacity = 10000

env = gym.make("CartPole-v0")
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

memory = replay_memory(capacity, num_states)
model = generate_model(num_states, num_actions)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

#!!!!
#episode_10_list
#!!!!

num_completes = 0
final_episode = False
frames = []
fig = plt.figure()

for episode in range(num_episodes):

    state = env.reset()

    for step in range(max_steps):

        if final_episode:
            img = env.render(mode="rgb_array")
            frames += [[plt.imshow(img)]]

        action = decide_action(model, state, episode, num_actions)

        state_n, _, done, _ = env.step(action)

        if done:
            if step < 195:
                reward = -1.0
                num_completes = 0
            else:
                reward = 1.0
                num_completes += 1
        else:
            reward = 0.0

        memory.push(state, action, state_n, reward)

        replay(memory, model, batch_size, optimizer, gamma)

        state = state_n

        if done:
            print("Episode {}: finished after {} time steps".format(episode, step + 1))
            break

    if final_episode:
        anim = animation.ArtistAnimation(fig, frames, interval=100)
        anim.save("anim_dqn.gif", writer="pillow", fps=30)
        break

    if num_completes >= 10:
        print("10 consecutive successes")
        final_episode = True
