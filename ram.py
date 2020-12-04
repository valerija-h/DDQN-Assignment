import tensorflow.compat.v1 as tf
import numpy as np
import random
import os
import matplotlib
import matplotlib.pyplot as plt
import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.optimizers import Adam
from collections import deque

"""
LOADING AND OBSERVING THE ENVIRONMENT
"""
# load the environment (that uses pixel images)
env_name = "Seaquest-ram-v0"
env = gym.make(env_name)

# examine the observation space and action space
print("Observation space: {}".format(env.observation_space))
print("Action space: {}".format(env.action_space))

# run a random action - to view info
env.reset()
next_obs, reward, done, info = env.step(0)
print('Info: {}'.format(info))

"""
CREATING THE AGENT
"""
class QLearningAgent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        self.learning_rate = 0.001
        self.discount_rate = 0.99
        self.checkpoint_path = "./pixel_seaquest_ram.ckpt"  # where to save model checkpoints
        self.min_epsilon = 0.1  # make sure it will never go below 0.1
        self.max_epsilon = 0.999
        self.loss_val = np.infty  # initialize loss_val

        tf.reset_default_graph()
        tf.disable_eager_execution()

        # observation variable - takes 128
        self.X_state = tf.placeholder(tf.float32, shape=[None, 128])
        # create two deep neural network - one for main model one for target model
        self.main_q_values, self.main_vars = self.create_model(self.X_state, name="main")  # main learns from target then target gets updated to main
        self.target_q_values, self.target_vars = self.create_model(self.X_state, name="target")  # we will use the main network to update this one

        # update the target network to have same weights of the main network
        # loop through each item in 'target_vars' and grab a list of the values we are going to change - this is the operations list
        self.copy_ops = [targ_var.assign(self.main_vars[targ_name]) for targ_name, targ_var in self.target_vars.items()]
        self.copy_online_to_target = tf.group(*self.copy_ops)  # group to apply the operations list

        # we create the model for training
        with tf.variable_scope("train"):
            # variables for actions (X_action) and target values (y)
            self.X_action = tf.placeholder(tf.int32, shape=[None])
            self.y = tf.placeholder(tf.float32, shape=[None, 1])

            # TODO - QnA session - vector q values * one hot encoding and obtain ???
            self.q_value = tf.reduce_sum(self.main_q_values * tf.one_hot(self.X_action, self.action_size),
                                         axis=1, keepdims=True)

            # used to make the target of q table close to real value
            # usually we just square loss but if we square it on its own, it will explode, so instead we will multiply loss by 2 which is above 1
            self.error = tf.abs(self.y - self.q_value)
            self.clipped_error = tf.clip_by_value(self.error, 0.0, 1.0)  # clip the value, if it is above 1 it stays at 1
            self.linear_error = 2 * (self.error - self.clipped_error)  # avoid exploding losses
            self.loss = tf.reduce_mean(tf.square(self.clipped_error) + self.linear_error)
            # an alternative to above would just be error squared - to avoid exploiding we use linear error and clipping (this is an optimization)

            # global step to remember the number of times the optimizer was used
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # to take the optimizer and tell it to minimize the loss, the function will also add +1 to global_step at each iteration
            self.training_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # saving the session - if u close the notebook it will load back the previous model
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        if os.path.isfile(self.checkpoint_path + ".index"):
            self.saver.restore(self.sess, self.checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.copy_online_to_target)

    """
    CREATING THE CNN NETWORK
    """
    def create_model(self, X_state, name):
        prev_layer = X_state / 255.0  # scale pixel intensities to the [-1.0, 1.0] range.
        initializer = tf.variance_scaling_initializer()

        with tf.variable_scope(name) as scope:
            prev_layer = tf.layers.dense(prev_layer, 24, activation=tf.nn.relu, kernel_initializer=initializer)
            prev_layer = tf.layers.dense(prev_layer, 24, activation=tf.nn.relu, kernel_initializer=initializer)
            output = tf.layers.dense(prev_layer, self.action_size, kernel_initializer=initializer)

        # create a dictionary of trainable vars by their name
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
        return output, trainable_vars_by_name

    """ ------- CHOOSING AN ACTION -------"""
    def get_action(self, state):
        q_values = self.main_q_values.eval(feed_dict={self.X_state: [state]})
        epsilon = max(self.min_epsilon, self.max_epsilon * self.global_step.eval())  # slowly decrease epsilon based on experience (global_step)

        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)  # choose random action
        else:
            return np.argmax(q_values)  # optimal action

    """ ------- TRAINING -------"""
    def train(self, state, action, reward, next_state, done):
        # compute q values of next state
        next_q_values = self.target_q_values.eval(feed_dict={self.X_state: np.array([next_state])})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)  # get q values with best rewards

        # compute target values
        y_val = reward + (done * (self.discount_rate * max_next_q_values))
        # if done is true (0), if he loses, y_val = only reward

        # train the main network
        _, self.loss_val = self.sess.run([self.training_op, self.loss], feed_dict={self.X_state: np.array([state]),
                                                                                   self.X_action: np.array([action]),
                                                                                   self.y: y_val})

agent = QLearningAgent(env)
train_steps = 10000  # total number of training steps in an episode
episodes = 1000  # number of episodes
copy_steps = 500  # update target network (from main network) every n steps
save_steps = 1000  # save model every n steps

with agent.sess:
    for e in range(episodes):
        state = env.reset()
        done = False
        for t in range(train_steps):

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state

            agent.train(state, action, reward, next_state, 1.0 - done)  # pass in inverse of done
            env.render()

            state = next_state

            # display training progress
            print("\r\tEpisode {}/{}\tStep {}/{} ({:.1f})%\tLoss {:5f}".format(e+1, episodes,
                                                                               t, train_steps, t*100/train_steps,
                                                                               agent.loss_val))
            # go to next episode
            if done: break

            # regulary update target DQN - every few steps
            if t % copy_steps == 0:
                agent.copy_online_to_target.run()

            # save model regularly - every few steps
            if t % save_steps == 0:
                agent.saver.save(agent.sess, agent.checkpoint_path)