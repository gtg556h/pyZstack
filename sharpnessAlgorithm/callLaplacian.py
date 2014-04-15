# Script to try and validate focus detection algorithms

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import scipy.signal
from PIL import Image
import zstackLib
import time

plt.ion()

filename = '/home/brian/git/pyZstack/lena.tif'
image = Image.open(filename)
image = np.array(image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blurRate = 10
j=1
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sharpness = []
while j <= 10:
    image2 = cv2.GaussianBlur(image, (j*blurRate-1,j*blurRate-1),0)
    sharp, conv = zstackLib.sharpnessLaplace(image2)
    sharpness.append(sharp)

    if 0:
        ax1.imshow(image2,'gray')
        ax2.hist(image2.flatten(), 256, range=(0,256), fc='k', ec='k', normed=True)
        ax2.set_xlim([0,256])
        ax2.set_ylim([0,.018])
        ax3.hist(conv.flatten(), 256-40, range=(40,256), fc='k', ec='k', normed=True)
        ax3.set_xlim([40,256])
        ax3.set_ylim([0,.030])
        plt.show()
        plt.pause(0.001)
        plt.waitforbuttonpress() 
        ax1.cla()
        ax2.cla()
        ax3.cla()

    j += 1

sharpness = np.asarray(sharpness)

plt.plot(sharpness)
plt.show()
plt.pause(0.001)
plt.waitforbuttonpress()
plt.close()
#plt.hist(image.flatten(), 256, range=(0,256), fc='k', ec='k')
