##############################################
#  Library of tools for z-stack analysis     #
#  Brian Williams                            #
#  2014.03.25                                #
#  brn.j.williams@gmail.com                  #
##############################################


from __future__ import division
import zstackLib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy
import cv2
from scipy import ndimage
import scipy.signal

###############################
###############################

class zstack(object):
    def __init__(self, filename):
        self.im = Image.open(filename)
        self.countFrames()


    ##########################################################
    def countFrames(self):
        i=0
        while 1:
            try:
                self.im.seek(i)
            except EOFError:
                self.nFrames = i
                break
            i+=1
                
    ###########################################################
    def focusScan(self, threshValue=127, maxValue=255, adaptive=0):
        self.totalSize = []
        self.nParticles = []
        self.sharpness = []
        j = 0
        while j<40: #1: 
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

            self.sharpness.append(zstackLib.sharpnessLaplace(gray))

        self.totalSize = np.asarray(self.totalSize)
        self.nParticles = np.asarray(self.nParticles)
        self.sharpness = np.asarray(self.sharpness)


    #######################################################        
    def scanTemplate(self):
        i = 0
        while 1:
            try: 
                self.im.seek(i)
                print(i)
            except EOFError:
                print(i, " frames total")
                break

            i+=1

    ######################################################

    ######################################################
    def sharpnessFourier(self):
        # High frequency content indicates sharpeness
        print('write some code')



    ######################################################
    def writeImage(self,filename,im):
        scipy.misc.imsave(filename,im)


##########################################
def sharpnessLaplace(image):
    # Convolve image w/ laplacian kernel: 
    #    1
    # 1 -4  1
    #    1

    laplaceKernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    image2 = np.zeros([image.shape[0]+4,image.shape[1]+4])
    image2[2:image.shape[0]+2,2:image.shape[1]+2] = image
    for i in range(2,image.shape[0]+2):
        image2[i,0] = image2[i,2]
        image2[i,1] = image2[i,2]
        image2[i,image2.shape[1]-2] = image2[i,image.shape[1]-3]
        image2[i,image2.shape[1]-1] = image2[i,image.shape[1]-3]
    for i in range(2,image.shape[1]+2):
        image2[0,i] = image2[2,i]
        image2[1,i] = image2[2,i]
        image2[image2.shape[0]-2,i] = image2[image2.shape[0]-3,i]
        image2[image2.shape[0]-1,i] = image2[image2.shape[0]-3,i]

    conv = np.abs(scipy.signal.convolve2d(image2,laplaceKernel))
    return np.sum(conv)



