import pickle


import numpy as np

import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import os
import csv
import sklearn
import cv2

import data_utils as du
import image_utils as iu
import importlib

importlib.reload(du)
importlib.reload(iu)


#%%

vs = du.loadDataFrames(['data_5'])

#train_samples = train_samples.query('angle!=0')
#validation_samples = validation_samples.query('angle!=0 ')

#%%
     
import scipy.signal

vs = du.loadDataFrames(['data_track_2_right_1'])
vs['anglef'] = 1.3*scipy.signal.savgol_filter(vs.angle,window_length=41,polyorder=2)
vs.plot(y=['angle', 'anglef'], figsize=(15,5))
vs.plot(y=['angle'], figsize=(35,5))
vs.plot(y=['anglef'], figsize=(35,5))
"""
vs = du.loadDataFrames(['data_4'])
vs.plot(y='angle', figsize=(15,5))
"""
plt.show()
print("hi")



#%%


pdf1 = du.loadDataFrames(['data_1'])
pdf2 = du.loadDataFrames(['data_4'])


plt.figure(figsize=(12,12))
plt.hist(pdf1.query('angle>0.05 | angle<-0.1 ').angle, bins=100)
plt.show()


#%%

#Compare PIL and CV2 pipelines

from PIL import Image

im = Image.open(pdf1.imgcenter[0])
plt.figure()
plt.imshow(im)

im2 = iu.cropAndResize(iu.PILimage2CV2HSV(im))
plt.figure()
plt.imshow(im2)

img3 = iu.cropAndResize(iu.imgload2HSV(pdf1.imgcenter[0]))
plt.figure()
plt.imshow(img3)

#%%


#Show one image in different color spaces.

pdf1 = du.loadDataFrames(['data_1'])
pdf2 = du.loadDataFrames(['data_4'])


def printImageComponents(image, titles):
    fig = plt.figure(figsize=(14,10))
    for i in [0,1,2]:
        plt.subplot(1,3,i+1)
        plt.axis('off')
        plt.imshow(image[:,:,i], cmap='gray')
        plt.title(titles[i])
        
def printImageComparisons(img):
    plt.figure()
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    printImageComponents(img, ['B','G','R'])
    printImageComponents(cv2.cvtColor(img,cv2.COLOR_BGR2YUV), ['Y','U','V'])
    printImageComponents(cv2.cvtColor(img,cv2.COLOR_BGR2HSV), ['H','S','V'])


img = cv2.imread(pdf1.imgcenter[0])
printImageComparisons(img)

img = cv2.imread(pdf1.imgcenter[1000])
printImageComparisons(img)

img = cv2.imread(pdf2.imgcenter[0])
printImageComparisons(img)

img = cv2.imread(pdf2.imgcenter[1000])
printImageComparisons(img)

img = cv2.imread(pdf2.imgcenter[200])
printImageComparisons(img)

plt.show()    


#%%

#Read the csv file

csvfilename = './data/data_5/driving_log.csv'

samples=[]

angles = []
with open(csvfilename) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        angles.append(float(line[3]))
        

#%%


plt.hist(angles, bins=40)



#%%
imkey = 100
im = cv2.imread(samples[imkey][0])
im2 = im.copy()

print(angles[imkey])
#Flipping

plt.subplot(1,2,1)
plt.imshow(im)
plt.subplot(1,2,2)
plt.imshow(cv2.flip(im2,1))
plt.show()

#%%
def getflipped(image):
    newimg = image.copy()
    newimg = cv2.flip(newimg, 1)
    return newimg

def switchRGB(image):
    b,g,r = cv2.split(image)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])
    return rgb_img

def PILimage2YUV(image):
    i = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(i,cv2.COLOR_BGR2YUV)

def CV2image2YUV(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2YUV)

#%%

from PIL import Image

imm = Image.open(samples[imkey][0])
im_bgr = cv2.imread(samples[imkey][0])
im = switchRGB(im_bgr)

imm_yuv = PILimage2YUV(imm)
im_yuv= CV2image2YUV(im_bgr)
 
plt.figure(figsize=(14,7))

plt.subplot(2,2,1)
plt.imshow(imm)

plt.subplot(2,2,2)
plt.imshow(im)

plt.subplot(2,2,3)
plt.imshow(imm_yuv)

plt.subplot(2,2,4)
plt.imshow(getflipped(im_yuv))
plt.show()

print("end")


#%%

import data_utils as du
import importlib

importlib.reload(du)

train_samples = du.loadData(['data_1', 'data_2', 'data_3', 'data_4'])


#%%


pdf = du.loadDataFrames(['data_1', 'data_2', 'data_3', 'data_4'])

len(pdf.query('angle > 0'))
len(pdf.query('angle==0'))
