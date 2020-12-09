import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gym
import os
import seaborn as sns

# this file is just for plotting different graphs and figures

# --------------------- PLOTTING PREPROCESSING BEFORE AND AFTER -------------------------------
def prep_obs(obs):
    img = obs[1:192:2, ::2]
    img = img.mean(axis=2).astype(np.uint8)  # convert to grayscale (values between 0 and 255)
    return img.reshape(96, 80, 1)/255

def show_preprocessing():
    img = cv2.imread('images/sample.png')
    img = cv2.resize(img, dsize=(160, 210))

    # show current image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

    # show preprocessed image
    plt.imshow(prep_obs(img).reshape(96, 80), cmap='gray', vmin=0, vmax=1)
    plt.show()


# --------------------- PLOTTING RESULTS FOR EACH INDIVIDUAL MODEL ------------------------------
def show_result(filename, title, colour):
    data = pickle.load(open(filename, 'rb'))
    mean = sum(data)/len(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    # fig.suptitle(title, fontsize=16)
    ax.plot(data, colour)
    ax.plot([0, 500], [mean, mean], 'k:', lw=3, label='Mean Total Reward: %0.3f' % mean)
    ax.set_ylabel('Total Reward')
    ax.set_xlabel('Episodes')
    ax.xaxis.set_ticks(np.arange(0, 501, 20))
    ax.yaxis.set_ticks(np.arange(0, 22, 1))
    plt.legend(prop={'size': 12})
    plt.savefig('images/'+os.path.splitext(filename.split('/')[1])[0]+'.png')
    plt.show()

    return data

all_data = []
all_data.append(show_result('results/ram_seaquest_test.p', 'Total Reward per Episode (RAM model)', 'b'))
all_data.append(show_result('results/pixel_seaquest_test.p', 'Total Reward per Episode (Pixel model)', 'g'))
all_data.append(show_result('results/pixel_ram_seaquest_test.p', 'Total Reward per Episode (Pixel and RAM model)', 'm'))

# --------------------- PLOTTING RESULTS FOR ALL MODELS ------------------------------
def get_avg(data):
    return sum(data)/len(data)

def show_all_results(all_data):
    fig, ax = plt.subplots(figsize=(15, 5))
    # fig.suptitle('Total Reward per Episode (all models)', fontsize=16)
    ax.plot(all_data[0], '-bo', lw=1, markersize=1, label='RAM - Mean Total Reward: %0.3f' % get_avg(all_data[0]))
    ax.plot(all_data[1], '-go', lw=1, markersize=1, label='Pixel - Mean Total Reward: %0.3f' % get_avg(all_data[1]))
    ax.plot(all_data[2], '-mo', lw=1, markersize=1, label='Pixel and RAM - Mean Total Reward: %0.3f' % get_avg(all_data[2]))
    # ax.plot([0, 500], [mean, mean], 'k:', lw=3, label='Mean Total Reward')
    ax.set_ylabel('Total Reward')
    ax.set_xlabel('Episodes')
    ax.xaxis.set_ticks(np.arange(0, 501, 20))
    ax.yaxis.set_ticks(np.arange(0, 22, 1))
    plt.legend(prop={'size': 12})
    plt.savefig('images/all_models.png')
    plt.show()

show_all_results(all_data)









