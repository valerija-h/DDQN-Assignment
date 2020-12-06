import cv2
import matplotlib.pyplot as plt
import numpy as np
import gym

def prep_obs(obs):
    img = obs[1:192:2, ::2]
    img = img.mean(axis=2).astype(np.uint8)  # convert to grayscale (values between 0 and 255)
    return img.reshape(96, 80, 1)/255

img = cv2.imread('images/sample.png')
img = cv2.resize(img, dsize=(160, 210))

# show current image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# show preprocessed image
plt.imshow(prep_obs(img).reshape(96, 80), cmap='gray', vmin=0, vmax=1)
plt.show()

