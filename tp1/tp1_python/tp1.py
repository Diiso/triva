import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

Irgb = mpimg.imread('images/tools.jpg')
plt.axis('off')
imgplot = plt.imshow(Irgb)

if not os.path.exists("results"):
    os.makedirs("results")
    
I = rgb2gray(Irgb)
fig = plt.figure(figsize=(15,15))

fig.add_subplot(1,2,1)
plt.axis('off')
plt.imshow(Irgb)

fig.add_subplot(1,2,2)
plt.axis('off')
plt.imshow(I, cmap = plt.get_cmap('gray'))

fig.savefig('results/black_and_white.jpg')

sigma_noise = 30
n,m = I.shape
I_noise = I + np.random.randn(n,m)*sigma_noise

fig = plt.figure(figsize=(15,15))

fig.add_subplot(1,2,1)
plt.axis('off')
plt.imshow(I, cmap=plt.get_cmap('gray'))

fig.add_subplot(1,2,2)
plt.axis('off')
plt.imshow(I_noise, cmap=plt.get_cmap('gray'))

fig.savefig('results/noise.jpg')

import gaussian_convolution as gc
I_blurred = gc.gaussianConvolution(I_noise,3)