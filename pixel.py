import tensorflow.compat.v1 as tf
import os
import matplotlib.pyplot as plt
import gym
import numpy as np
from collections import deque
from IPython.display import clear_output
import random
import time

"""
LOADING AND OBSERVING THE ENVIRONMENT
"""
# load the environment (that uses pixel images)
env_name = "Seaquest-v0"
env = gym.make(env_name)

# examine the observation space and action space
print("Observation space: {}".format(env.observation_space))
print("Action space: {}".format(env.action_space))

# run a random action - to view info
env.reset()
next_obs, reward, done, info = env.step(0)
print('Info: {}'.format(info))

"""
PREPROCESSING THE OBSERVATIONS
"""
def prep_obs(obs):
    img = obs[1:192:2, ::2]
    img = img.mean(axis=2)  # convert to grayscale (values between 0 and 255)
    return img.reshape(96, 80, 1)

obs = env.reset()
plt.imshow(obs)
plt.show()
plt.imshow(prep_obs(obs).reshape(96,80), cmap='gray', vmin=0, vmax=255)
plt.show()

"""
CREATING THE AGENT
"""
class QLearningAgent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        self.observation_size = (96, 80, 1)
        self.learning_rate = 0.001 # higher for experience replay
        self.discount_rate = 0.99
        self.checkpoint_path = "./pixel_seaquest_test.ckpt"  # where to save model checkpoints
        self.min_epsilon = 0.1  # make sure it will never go below 0.1
        self.max_epsilon = 0.999
        self.loss_val = np.infty  # initialize loss_val
        self.replay_buffer = deque(maxlen=1000)  # exerience buffe

        tf.reset_default_graph()
        tf.disable_eager_execution()

        # observation variable - takes shape 96 by 80
        self.X_state = tf.placeholder(tf.float32, shape=[None, 96, 80, 1])
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
            self.y = tf.placeholder(tf.float32, shape=[None])

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
            prev_layer = tf.layers.conv2d(prev_layer, filters=32, kernel_size=8, strides=4, padding="SAME",
                                          activation=tf.nn.relu, kernel_initializer=initializer)
            prev_layer = tf.layers.conv2d(prev_layer, filters=64, kernel_size=4, strides=2, padding="SAME",
                                          activation=tf.nn.relu, kernel_initializer=initializer)
            prev_layer = tf.layers.conv2d(prev_layer, filters=64, kernel_size=3, strides=1, padding="SAME",
                                          activation=tf.nn.relu, kernel_initializer=initializer)
            flatten = tf.reshape(prev_layer, shape=[-1, 64 * 12 * 10])
            hidden = tf.layers.dense(flatten, 512, activation=tf.nn.relu, kernel_initializer=initializer)
            output = tf.layers.dense(hidden, self.action_size, kernel_initializer=initializer)

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
    def train(self, experience, batch_size=32):
        self.replay_buffer.append(experience)  # add experience to buffer

        # extract an experience batch from the buffer
        samples = random.choices(self.replay_buffer, k=batch_size)
        state, action, next_state, reward, done = (list(col) for col in zip(experience, *samples))

        # compute q values of next state
        next_q_values = self.target_q_values.eval(feed_dict={self.X_state: np.array(next_state)})
        next_q_values[done] = np.zeros([self.action_size])  # set to 0 if done = true

        # compute target values
        y_val = reward + self.discount_rate * np.max(next_q_values)

        # train the main network
        _, self.loss_val = self.sess.run([self.training_op, self.loss], feed_dict={self.X_state: np.array(state),
                                                                                   self.X_action: np.array(action),
                                                                                   self.y: y_val})


agent = QLearningAgent(env)
episodes = 10000  # number of episodes
list_rewards = []
total_reward = 0  # reward per episode
copy_steps = 500  # update target network (from main network) every n steps
save_steps = 1000  # save model every n ste

with agent.sess:
    for e in range(episodes):
        state = prep_obs(env.reset())
        done = False
        list_rewards.append(total_reward)
        total_reward = 0
        while not done:
            step = agent.global_step.eval()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = prep_obs(next_state)

            agent.train((state, action, next_state, reward, done))  # pass in inverse of done
            env.render()
            state = next_state
            total_reward += reward

            print("\r\tEpisode: {}/{},\tStep: {}\tTotal Reward: {},\tLoss: {}".format(e+1, episodes, step, total_reward, agent.loss_val))

            time.sleep(0.01)
            clear_output(wait=True)

            # regulary update target DQN - every n steps
            if step % copy_steps == 0:
                agent.copy_online_to_target.run()

            # save model regularly - every n steps
            if step % save_steps == 0:
                agent.saver.save(agent.sess, agent.checkpoint_path)
