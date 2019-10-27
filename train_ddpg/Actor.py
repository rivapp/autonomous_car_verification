"""
This class implements the policy neural network. 
"""

import tensorflow as tf
import math

from keras import models
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Lambda
from keras.optimizers import Adam
from keras.constraints import max_norm

# Actor/Critic Neural Network Architecture
LAYER_1_SIZE = 400
LAYER_2_SIZE = 300

class ActorNetwork(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, layer1_size = LAYER_1_SIZE, layer2_size = LAYER_2_SIZE):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size

        # Create non_target policy network
        self.inputs, self.out, self.scaled_out = self.create_actor_network(nn_type = "non_target", name='non-target')

        # Find trainable variables that will be needed for gradient calculations
        self.network_params = tf.trainable_variables()
        print(self.network_params[0].get_shape().as_list())


        # Create target policy network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network(nn_type = "target", name='target')

        # Find trainable variables that will be needed for gradient calculations
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]


        # Below are some check that the trainable weights have the correct shape
        for i in range(len(self.target_network_params)):
            print(i)
            elem_1 = tf.multiply(self.network_params[i],self.tau)
            print("elem_1",elem_1.get_shape().as_list())
            elem_2 = tf.multiply(self.target_network_params[i],1.-self.tau)
            print("elem_2",elem_2.get_shape().as_list())
            tensor_sum = elem_1 + elem_2

        print("act_nn shape",elem_1)
        print("target_nn shape",elem_2)
        tensor_sum = (elem_1 + elem_2).get_shape().as_list()
        print("Sum shape",tensor_sum)

        # Create operation in graph to update target neural network
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau)) for i in range(len(self.target_network_params))]
        print("length",len(self.update_target_network_params))

        # Store critic gradient with respect to actions.  To be used in policy gradient update.
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Determine gradients of actions with respect to parameters of actor network.  Combine gradients of Q-value with respect
        # to action and action with respect to parameters of neural network.  The produce is negated to perform gradient ascent
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        self.optimize = tf.keras.optimizers.Adam(self.learning_rate, clipnorm=1).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)


    # Create policy network with architecture provided.  The policy network is a mapping from states to actions.
    def create_actor_network(self, nn_type, name):

        # States are the input variable
        states = Input(shape = (self.s_dim,) + (1,))
        flattened = Flatten()(states)

        h1 = Dense(units = self.layer1_size, activation='tanh', name=name + 'hl1', kernel_constraint=max_norm(1), bias_constraint=max_norm(1))(flattened)

        h2 = Dense(units = self.layer2_size, activation='tanh', name=name + 'hl2', kernel_constraint=max_norm(1), bias_constraint=max_norm(1))(h1)
        
        act_not_scaled = Dense(units = self.a_dim, activation = 'tanh', name = name + 'ol')(h2)

        act_scaled = Lambda(lambda x: x * self.action_bound, name=name + 'ols')(act_not_scaled)
        
        return states, act_not_scaled, act_scaled


    # The train function is how the "Learning" phase of the learning loop is implemented.  The policy neural network
    # parameters are updated in the diretion of the deterministic policy gradient
    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    # The predict function is used to find the action prescribed by the policy network given the current state.
    # The agent will experiment by adding noise to the output of this function
    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    # This function updates the target neural network by using values from the non_target network
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
