from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy
import cv2
from scipy import ndimage


class zstack(object):
    def __init__(self, filename):
        self.im = Image.open(filename)
        self.countFrames()


    def countFrames(self):
        i=0
        while 1:
            try:
                self.im.seek(i)
            except EOFError:
                self.nFrames = i
                break
            i+=1
                

    def scan(self, threshValue=127, maxValue=255, adaptive=0):
        self.totalSize = []
        self.nParticles = []
        j = 0
        while 1: 
            try:
                self.im.seek(j)
                print(j)
            except EOFError:
                break

            image = np.array(self.im)

            try:
                image.shape[2]
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            except IndexError:
                gray = image

            gray = cv2.GaussianBlur(gray, (5,5), 0)
            gray = gray.clip(0,255)
            gray = gray.astype('B')
            threshValue = np.mean(gray) + 0.2*(np.max(gray)-np.min(gray))
            
            if adaptive == 1:
                print('code me')
            else:
                method = cv2.THRESH_BINARY
                ret, thresh = cv2.threshold(gray, threshValue, maxValue, method)

            labelarray, particle_count = ndimage.measurements.label(thresh)
            particleSize = np.zeros(particle_count)
            for i in range(0, particle_count):
                particleSize[i] = np.size(np.where(labelarray==i)[0])
            totalSize = np.sum(particleSize[1:particleSize.shape[0]])

            labelarray, particle_count = ndimage.measurements.label(thresh)
            self.nParticles.append(particle_count)
            particleSize = np.zeros(particle_count)

            for i in range(0, particle_count):
                particleSize[i] = np.size(np.where(labelarray==i)[0])
            self.totalSize.append(np.sum(particleSize[1:particleSize.shape[0]]))
            j+=1

		#self.totalSize = np.asarray(self.totalSize)
		#self.nParticles = np.asarray(self.nParticles)

            
    def scanTrial(self):
        i = 0
        while 1:
            try: 
                self.im.seek(i)
                print(i)
            except EOFError:
                print(i, " frames total")
                break

            i+=1


    def writeImage(self,filename,im):
        scipy.misc.imsave(filename,im)
