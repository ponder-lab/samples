#!/usr/bin/env python

import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import slime_env

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()

    # !!!!!!!! Need to reset when episode ends !!!!!!!!
    end_of_episode = False
    if time_step.is_last():
        end_of_episode = True

    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)

    # !!!!!!!! Need to reset when episode ends !!!!!!!!
    if end_of_episode:
        environment.reset()
    
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

def eval_show(eval_env, eval_py_env, agent):
    time_step = eval_env.reset()
    img = eval_py_env.render()
    while not time_step.is_last():
        action_step = agent.policy.action(time_step)
        #action_step = agent.collect_policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        img = eval_py_env.render()
    eval_py_env.close()

# collect_data(train_env, random_policy, replay_buffer, steps=100)

# This loop is so common in RL, that we provide standard implementations. 
# For more details see the drivers module.
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/drivers

#seed = 2
#random.seed(seed)
#np.random.seed(seed)
#tf.random.set_seed(seed)

#
# Hyper parameters
#
#num_iterations = 20000
num_iterations = 600000

initial_collect_steps = 1000
collect_steps_per_iteration = 1
#collect_steps_per_iteration = 10
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-3
#learning_rate = 1e-4
#log_interval = 200
log_interval = 1000

num_eval_episodes = 10
#eval_interval = 1000
eval_interval = 50000

show_video = False
#show_video = True
show_interval = 50000

epsilon_greedy = 0.8

#
# Environment
#
#env_name = "CartPole-v0"
#env_name = "SlimeVolley-v0"
#train_py_env = suite_gym.load(env_name)
#eval_py_env = suite_gym.load(env_name)

train_py_env = slime_env.slime_env()
eval_py_env = slime_env.slime_env()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (100,)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

#optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate)

#train_step_counter = tf.Variable(0)
global_step = tf.compat.v1.train.get_or_create_global_step()

agent = dqn_agent.DqnAgent(
#agent = dqn_agent.DdqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    #train_step_counter=train_step_counter,
    train_step_counter=global_step,
    epsilon_greedy=epsilon_greedy)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

collect_data(train_env, random_policy, replay_buffer, steps=100)
#collect_data(train_env, agent.collect_policy, replay_buffer, steps=100)

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

train_checkpointer = common.Checkpointer(
    ckpt_dir="ckpt",
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    #global_step=train_step_counter
    global_step=global_step
)

train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()

# Reset the train step
#agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

if show_video:
    eval_show(eval_env, eval_py_env, agent)

for _ in range(num_iterations):
    
    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, agent.collect_policy, replay_buffer)
        #collect_step(train_env, random_policy, replay_buffer)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

    if show_video:
        if step % show_interval == 0:
            eval_show(eval_env, eval_py_env, agent)

# iterations = range(0, num_iterations + 1, eval_interval)
# plt.plot(iterations, returns)
# plt.ylabel('Average Return')
# plt.xlabel('Iterations')
# #plt.ylim(top=250)
# plt.show()

if show_video:
    for _ in range(5):
        eval_show(eval_env, eval_py_env, agent)
    
#train_checkpointer.save(train_step_counter)
train_checkpointer.save(global_step)
