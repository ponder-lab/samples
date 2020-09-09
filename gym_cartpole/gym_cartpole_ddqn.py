#!/usr/bin/env python

"""
DDQN code in the RL book ported to tensorflow
"""

import random
import numpy as np
import gym

import numpy as np
import tensorflow as tf

def generate_model(num_in, num_mid, num_out):
    input_node = tf.keras.Input(num_in)
    x = input_node
    x = tf.keras.layers.Dense(num_mid, activation="relu")(x)
    x = tf.keras.layers.Dense(num_mid, activation="relu")(x)
    x = tf.keras.layers.Dense(num_out)(x)
    output_node = x
    model = tf.keras.models.Model(inputs=input_node, outputs=output_node)
    return model

class replay_memory:
    def __init__(self, capacity, num_states):
        self.capacity = capacity
        self.state = np.empty((0, num_states), np.float32)
        self.action = np.empty((0,), np.int64)
        self.state_n = np.empty((0, num_states), np.float32)
        self.reward = np.empty((0,), np.float32)
        self.index = 0

    def push(self, state, action, state_n, reward):
        if len(self.state) < self.capacity:
            self.state = np.append(self.state, np.array([state], dtype=np.float32), axis=0)
            self.action = np.append(self.action, np.array([action], dtype=np.int64), axis=0)
            self.state_n = np.append(self.state_n, np.array([state_n], dtype=np.float32), axis=0)
            self.reward = np.append(self.reward, np.array([reward], dtype=np.float32), axis=0)
        else:
            self.state[self.index] = state
            self.action[self.index] = action
            self.state_n[self.index] = state_n
            self.reward[self.index] = reward

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        rand_idx = np.random.randint(self.state.shape[0], size=batch_size)
        state_batch = self.state[rand_idx, :]
        action_batch = self.action[rand_idx]
        state_n_batch = self.state_n[rand_idx, :]
        reward_batch = self.reward[rand_idx]

        return state_batch, action_batch, state_n_batch, reward_batch

    def __len__(self):
        return len(self.state)

class rl_agent:

    def __init__(self, num_states, num_actions, batch_size, capacity, gamma):

        self.num_actions = num_actions

        self.batch_size = batch_size
        self.memory = replay_memory(capacity, num_states)

        num_mid = 32
        self.main_q_network = generate_model(num_states, num_mid, num_actions)
        self.target_q_network = generate_model(num_states, num_mid, num_actions)
        self.main_q_network.summary()

        self.optimizer = tf.keras.optimizers.Adam(0.0001)

        self.gamma = gamma

    def get_action(self, state, episode):

        epsilon = 0.5 * (1.0 / (episode + 1))

        if epsilon < np.random.uniform(0, 1):
            action = self.main_q_network(state[None, ...])
            action = tf.argmax(action, axis=1).numpy()[0]
        else:
            action = np.random.randint(self.num_actions)

        return action

    def memorize(self, state, action, state_next, reward):
        self.memory.push(state, action, state_next, reward)

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        (state_batch, action_batch,
         state_n_batch, reward_batch) = self.memory.sample(self.batch_size)

        # Actions for next state by MAIN Q-Network
        tmp_next_q_values = self.main_q_network(state_n_batch)
        tmp_actions = tf.argmax(tmp_next_q_values, axis=1).numpy()

        # Q value of the next state by TARGET Q-Network
        select_idx = np.hstack([np.arange(batch_size).reshape(batch_size, 1),
                                tmp_actions.reshape(batch_size, 1)])
        next_q_values = self.target_q_network(state_n_batch)
        next_state_action_values = tf.gather_nd(next_q_values, select_idx)

        # Expected values
        expected_state_action_values = reward_batch + self.gamma * next_state_action_values

        # indices to select by action_batch
        select_idx = np.hstack([np.arange(batch_size).reshape(batch_size, 1),
                                action_batch.reshape(batch_size, 1)])

        with tf.GradientTape() as tape:
            # Q value of the current state for the selected action
            tmp_q_values = self.main_q_network(state_batch)
            state_action_values = tf.gather_nd(tmp_q_values, select_idx)
            loss = tf.keras.losses.Huber()(expected_state_action_values, state_action_values)

        gradients = tape.gradient(loss, self.main_q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.main_q_network.trainable_variables))

    def update_target_q_network(self):
        self.target_q_network.set_weights(self.main_q_network.get_weights())
    
#
# Parameters
#
env_name = 'CartPole-v0'
gamma = 0.99
max_step = 200
num_episodes = 500

batch_size = 32
capacity = 10000

env = gym.make(env_name)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

agent = rl_agent(num_states, num_actions, batch_size, capacity, gamma)

episode_10_list = np.zeros(10) # list of consecutive steps
complete_episodes = 0
episode_final = False
frames = []

for episode in range(num_episodes):

    observation = env.reset()

    state = observation

    for step in range(max_step):

        action = agent.get_action(state, episode)

        observation_next, _, done, _ = env.step(action)

        if done:
            episode_10_list = np.hstack((episode_10_list[1:], step+1))

            if step < max_step - 5:
                reward = -1.0
                complete_episodes = 0
            else:
                reward = 1.0
                complete_episodes += 1
        else:
            reward = 0.0

        state_next = observation_next

        agent.memorize(state, action, state_next, reward)
        agent.replay()

        state = state_next

        if done:
            print("%d episode: Finished after %d steps: average steps=%.1f" %
                  (episode, step + 1, episode_10_list.mean()))

            if episode % 2 == 0:
                agent.update_target_q_network()
            break

    if episode_final is True:
        break

    if complete_episodes >= 10:
        print("10 consecutive success!")
        episode_final = True

state = env.reset()
for step in range(max_step):
    action = agent.get_action(state, episode)
    state_next, _, _, _ = env.step(action)
    state = state_next
    env.render()
