"""
This is an implementation of the DDPG reiforcement learning algorithm.
"""

import tensorflow as tf
import numpy as np

# Import random noise generator class
from DDPG_Noise import OUNoise

# Import Replay Buffer class and deque data structure
import random
from memory import ReplayBuffer

# Import Actor and Critic network classes
from Actor import ActorNetwork
from Critic import CriticNetwork

import sys
sys.path.append('../simulator')
#Import environment
from Car import World
import gym
from gym import spaces

import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import models
        
# Learning Parameters

# Restore Variable used to load weights
RESTORE = False

# Number of episodes to be run
MAX_EPISODES = 2000

# Max number of steps in each episode
MAX_EP_STEPS = 130

# Layer sizes (these are currently number of filters)
l1size = 64
l2size = 64

# Learning rates
MAX_ACTOR_LEARNING_RATE = 1e-3
MIN_ACTOR_LEARNING_RATE = 1e-4

MAX_CRITIC_LEARNING_RATE = 5e-4
MIN_CRITIC_LEARNING_RATE = 5e-4

LR_CYCLE = 100

BUFFER_SIZE = 1000000

MINIBATCH_SIZE = 128

EXPLORATION_SIZE = 200

# Memory warmup
MEMORY_WARMUP = 1024

# Discount factor reflects the agents preference for short-term rewards over long-term rewards
GAMMA = 0.99

# Tau reflects how quickly target networks should be updated
TAU = 0.001

RANDOM_SEED = 25

CONST_THROTTLE = 16

LIDAR_FIELD_OF_VIEW = 115
LIDAR_NUM_RAYS = 21
LIDAR_NOISE = 0.1 #m
LIDAR_MISSING_RAYS = 0

modelfile = 'tanh' + str(l1size) + 'x' + str(l2size) +\
            '_' + str(LIDAR_NUM_RAYS) + '_missing_' + str(LIDAR_MISSING_RAYS) + '.h5'

def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread

# The train function implements the two-step learning cycle.
def train(sess, env, actor, critic, RESTORE):

    sess.run(tf.global_variables_initializer())
    
    # Initialize random noise generator
    exploration_noise = OUNoise(env.action_space.shape[0])
    
    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay buffER
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    totSteps = 0
    
    # Store q values for illustration purposes
    q_max_array = []

    actor.learning_rate = MAX_ACTOR_LEARNING_RATE
    critic.learning_rate = MAX_CRITIC_LEARNING_RATE

    for i in xrange(MAX_EPISODES):

        s = env.reset()
        s = normalize(s)

        ep_reward = 0
        ep_ave_max_q = 0

        # update learning rates using cosine annealing
        T_cur = i % LR_CYCLE
        actor.learning_rate = MIN_ACTOR_LEARNING_RATE +\
                              0.5 * (MAX_ACTOR_LEARNING_RATE - MIN_ACTOR_LEARNING_RATE) * \
                              (1 + np.cos(np.pi * T_cur / LR_CYCLE))
        
        critic.learning_rate = MIN_CRITIC_LEARNING_RATE +\
                              0.5 * (MAX_CRITIC_LEARNING_RATE - MIN_CRITIC_LEARNING_RATE) * \
                              (1 + np.cos(np.pi * T_cur / LR_CYCLE))
            

        for j in xrange(MAX_EP_STEPS):

            totSteps += 1

            # Begin "Experimentation and Evaluation Phase"
            
            # Select next experimental action by adding noise to action prescribed by policy
            a = actor.predict(np.reshape(s, (1, actor.s_dim, 1)))

            # If in a testing episode, do not add noise
            if i < EXPLORATION_SIZE and not (i % 100 is 49 or i % 100 is 99):
                noise = exploration_noise.noise()
                a = a + noise

            # Constrain action
            a = np.clip(a, -15, 15)

            # Take step with experimental action
            s2, r, terminal, info = env.step(np.reshape(a.T,newshape=(env.action_space.shape[0],)), CONST_THROTTLE)

            #print("car pos: " + str(env.car_dist_s))
            #print("action: " + str(a))
            #print("reward: " + str(r))

            s2 = normalize(s2)

            # Add transition to replay buffer if not testing episode
            if i%100 is not 49 and i%100 is not 99:
                replay_buffer.add(np.reshape(s, (actor.s_dim, 1)), np.reshape(a, (actor.a_dim,)), r,
                                  terminal, np.reshape(s2, (actor.s_dim, 1)))

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if replay_buffer.size() > MEMORY_WARMUP:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                    # Find target estimate to use for updating the Q-function
                    
                    # Predict_traget function determines Q-value of next state
                    target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                    # Complete target estimate (R(t+1) + Q(s(t+1),a(t+1)))
                    y_i = []
                    for k in xrange(MINIBATCH_SIZE):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + GAMMA * target_q[k])
                            
                    # Perform gradient descent to update critic
                    predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                    ep_ave_max_q += np.amax(predicted_q_value, axis = 0)

                    # Perform "Learning" phase by moving policy parameters in direction of deterministic policy gradient
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])
                    
                    # Update target networks                    
                    actor.update_target_network()
                    critic.update_target_network()

            s = s2
            ep_reward += r

            # If episode is finished, print results
            if terminal:
                
                if i%100 is 49 or i%100 is 99:
                    print("Testing")

                    kmodel = Sequential()
                    actVars = []
                    for var in tf.trainable_variables():
                        if 'non-target' in str(var):
                            actVars.append(var)

                    kmodel.add(Dense(units=l1size, activation='tanh', weights = [sess.run(actVars[0]), sess.run(actVars[1])], input_dim=actor.s_dim))
                    kmodel.add(Dense(units=l2size, activation='tanh', weights = [sess.run(actVars[2]), sess.run(actVars[3])]))
                    kmodel.add(Dense(units=1, activation='tanh', weights = [sess.run(actVars[4]), sess.run(actVars[5])]))
                    optimizer = optimizers.RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06)
                    kmodel.compile(loss="mse", optimizer=optimizer)
                    kmodel.save(modelfile)

                else:
                    print("Training")                   

                print ('| Reward: %.2i' % int(ep_reward), " | Episode", i, '| Qmax: %.4f' % (ep_ave_max_q / float(j)))
                q_max_array.append(ep_ave_max_q / float(j))

                print('Finished in ' + str(j) + ' steps')
                
                break
                    
    plt.plot(q_max_array)
    plt.xlabel('Episode Number')
    plt.ylabel('Max Q-Value')
    plt.show()
    
    kmodel = Sequential()
    actVars = []
    for var in tf.trainable_variables():
        if 'non-target' in str(var):
            actVars.append(var)            

    kmodel.add(Dense(units=l1size, activation='tanh', weights = [sess.run(actVars[0]), sess.run(actVars[1])], input_dim=actor.s_dim))
    kmodel.add(Dense(units=l2size, activation='tanh', weights = [sess.run(actVars[2]), sess.run(actVars[3])]))
    kmodel.add(Dense(units=1, activation = 'tanh', weights = [sess.run(actVars[4]), sess.run(actVars[5])]))
    optimizer = optimizers.RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06)
    kmodel.compile(loss="mse", optimizer=optimizer)
    kmodel.summary()
    kmodel.save(modelfile)
                
# Begin program                
def main():

    numHalls = 4
    hallWidth = 1.5
    hallLength = 20
    turns = ['right', 'right', 'right', 'right']
    car_dist_s = hallWidth/2.0
    car_dist_f = hallLength/2.0
    car_heading = 0
    time_step = 0.1
    
    with tf.Session() as sess:

        env = World(numHalls, hallWidth, hallLength, turns,\
                    car_dist_s, car_dist_f, car_heading, MAX_EP_STEPS,\
                    time_step, LIDAR_FIELD_OF_VIEW, LIDAR_NUM_RAYS,\
                    lidar_noise = LIDAR_NOISE, lidar_missing_rays = LIDAR_MISSING_RAYS)

        #np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        # Check environment dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        # Build actor and critic networks
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             MAX_ACTOR_LEARNING_RATE, TAU,
                             layer1_size = l1size, layer2_size = l2size,)
        
        critic = CriticNetwork(sess, state_dim, action_dim,
                               MIN_CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
        

        train(sess, env, actor, critic,RESTORE)


main()


# In[ ]:




# In[ ]:



