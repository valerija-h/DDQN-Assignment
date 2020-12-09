import tensorflow.compat.v1 as tf
import os
import matplotlib.pyplot as plt
import gym
import numpy as np
from collections import deque
from IPython.display import clear_output
import random
import time
import pickle
# please note the code in the agent class was adapated from tutorial material
# please note the prioritized replay was adapted from class material

"""
LOADING AND OBSERVING THE ENVIRONMENT
"""
# load the environment (that uses pixel images) without stochastic frame skipping
env_name = "SeaquestNoFrameskip-v4"
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
    # cropping and sub-sampling
    img = obs[1:192:2, ::2]
    # convert to grayscale (values between 0 and 255) and convert to uint8
    img = img.mean(axis=2).astype(np.uint8)
    # reshape for tensorflow compatibility
    return img.reshape(96, 80, 1)

obs = env.reset()
# plt.imshow(obs)
# plt.show()
# plt.imshow(prep_obs(obs).reshape(96,80), cmap='gray', vmin=0, vmax=255)
# plt.show()

"""
PRIORITIZED REPLAY
"""
class PrioritizedReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

    # A new experience is given the maximum priority
    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1.0))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1 / (len(self.buffer) * probabilities)
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return map(list, zip(*samples)), importance, sample_indices

    def set_priorities(self, indices, errors, offset=0.001):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset


"""
CREATING THE AGENT
"""
class QLearningAgent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        self.observation_size = (96, 80, 1)
        self.learning_rate = 0.00025  # higher for experience replay
        self.discount_rate = 0.95
        self.checkpoint_path = "seaquest_pixel.ckpt"  # where to save model checkpoints
        self.min_epsilon = 0.1  # make sure it will never go below 0.1
        self.epsilon = self.max_epsilon = 1.0
        self.final_exploration_frame = 100000
        self.loss_val = np.infty  # initialize loss_val
        self.error_val = np.infty
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=100000)  # experience buffer
        self.tau = 0.001

        tf.reset_default_graph()
        tf.disable_eager_execution()

        # observation variable - takes shape 96 by 80
        self.X_state = tf.placeholder(tf.float32, shape=[None, 96, 80, 1])
        # create two deep neural network - one for main model one for target model
        self.main_q_values, self.main_vars = self.create_model(self.X_state, name="main")
        self.target_q_values, self.target_vars = self.create_model(self.X_state, name="target")

        # update the target network to have the same weights as the main network
        self.copy_ops_hard = [targ_var.assign(self.main_vars[targ_name]) for targ_name, targ_var in self.target_vars.items()]
        self.copy_ops_soft = [targ_var.assign(targ_var * (1. - self.tau) + self.main_vars[targ_name] * self.tau) for targ_name, targ_var in self.target_vars.items()]
        self.copy_online_to_target = tf.group(*self.copy_ops_hard)  # group to apply the operations list

        # we create the model for training
        with tf.variable_scope("train"):
            # variables for actions (X_action) and target values (y)
            self.X_action = tf.placeholder(tf.int32, shape=[None])
            self.y = tf.placeholder(tf.float32, shape=[None])
            self.importance = tf.placeholder(tf.float32, shape=[None])

            self.q_value = tf.reduce_sum(self.main_q_values * tf.one_hot(self.X_action, self.action_size), axis=1)

            # calculate error and loss function
            self.error = self.y - self.q_value
            self.loss = tf.reduce_mean(tf.multiply(tf.losses.huber_loss(self.y, self.q_value, reduction='none'), self.importance))

            # global step to remember the number of times the optimizer was used
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # tell optimizer to minimize loss, the function will also add +1 to global_step at each iteration
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
        prev_layer = X_state / 255.0  # scale pixel intensities to the [0, 1.0] range.
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

        # slowly decrease epsilon
        self.epsilon = max(self.min_epsilon, self.max_epsilon - ((self.max_epsilon - self.min_epsilon)/
                                                                 self.final_exploration_frame)*self.global_step.eval())

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)  # choose random action
        else:
            return np.argmax(q_values)  # optimal action

    """ ------- TRAINING -------"""
    def train(self, experience, batch_size=32, priority_scale=0.0):
        self.replay_buffer.add(experience)  # add experience to buffer

        # extract an experience batch from the buffer
        (state, action, next_state, reward, done), importance, indices = self.replay_buffer.sample(batch_size, priority_scale=priority_scale)

        # compute q values of next state
        next_q_values = self.target_q_values.eval(feed_dict={self.X_state: np.array(next_state)})
        next_q_values[done] = np.zeros([self.action_size])  # set to 0 if done = true

        # compute target values
        y_val = reward + self.discount_rate * np.max(next_q_values)

        # train the main network
        importance = (importance**(1-self.epsilon)).reshape((importance.shape[0],))
        feed = {self.X_state: np.array(state), self.X_action: np.array(action), self.y: y_val, self.importance: importance}
        _, self.loss_val, self.error_val = self.sess.run([self.training_op, self.loss, self.error], feed_dict=feed)
        self.replay_buffer.set_priorities(indices, self.error_val)


agent = QLearningAgent(env)
episodes = 500  # number of episodes
copy_steps = 10000  # update target network (from main network) every n steps
save_steps = 10000  # save model every n steps
list_rewards = []
frame_skip_rate = 4

with agent.sess:
    for e in range(episodes):
        state = prep_obs(env.reset())
        done = False
        total_reward = 0
        losses = []
        i = 1  # iterator to keep track of steps per episode - for frame skipping and avg loss
        action = 0  # do nothing at start - no states yet
        while not done:
            step = agent.global_step.eval()

            # get a new action every X frames (frame skipping)
            if i % frame_skip_rate == 0:
                action = agent.get_action(state)

            # get outcome of action - perform last action for X steps
            next_state, reward, done, info = env.step(action)
            next_state = prep_obs(next_state)
            reward = np.sign(reward)  # in reward clipping all positive rewards are +1 and all negative is -1

            # train model every X frames (frame skipping)
            if i % frame_skip_rate == 0:
                agent.train((state, action, next_state, reward, done), priority_scale=0.8)

            state = next_state
            total_reward += reward

            # regulary update target DQN - every n steps
            if step % copy_steps == 0:
                agent.copy_online_to_target.run()

            # save model regularly - every n steps
            if step % save_steps == 0:
                agent.saver.save(agent.sess, agent.checkpoint_path)

            i += 1
        print("\r\tEpisode: {}/{},\tStep: {}\tTotal Reward: {}".format(e + 1, episodes, step, total_reward))
        list_rewards.append(total_reward)
        pickle.dump(list_rewards, open("results/pixel_seaquest_test.p", "wb"))

plt.plot(list_rewards)
plt.show()
