import pickle
import tensorflow as tf

# Keras Layers needed
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D
from keras.layers import Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D
from keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import os
import csv
import sklearn
import cv2

# Utility functions for the project
import data_utils as du
import image_utils as iu

    
def generator(samples, batch_size=32):
    '''
    Generator used to feed the training process.  Instead of loading data directly we use a generator
    to preprocess the images through out the trainingImagePipeline
    :param samples: sample dataframe 
    :param batch_size: size of each batch to be processed from the samples.  Note than total number of
        images will be duplicated
    :return: The X and labels for the batch
    '''
    num_samples = len(samples)
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]
            images=[]
            angles=[]
            for i,batch_sample in batch_samples.iterrows():

                image = iu.trainingImagePipeline(batch_sample.imgfile)
                angle = batch_sample.angle
                
                images.append(image)
                angles.append(angle)
                
                ## Data augmentation, flipped image
                images.append(iu.getflipped(image))
                angles.append(-angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            

def getModel(learning_rate=0.001, dropout=0.3):
    '''
    Returns the model to be used.
    This model is a variation of the NVIDIA model from "End to End Learning for Self-Driving Cars"
    It contains 4 Convolutional Layers and 3 fully connected layers.
    :param learning_rate: learning rate to be used by the Adam optimizer
    :param dropout: dropout used in the 2 dropout layers
    :return: The Keras model.
    '''
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(64,96,3)))
    model.add(Convolution2D(12,4,4,activation='relu'))  #43z158 x 24
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(24,4,4,activation='relu',subsample=(2,2)))  #
    model.add(Convolution2D(36,4,4,activation='relu',subsample=(2,2)))  #
    model.add(Dropout(dropout))
    model.add(Convolution2D(64,4,4,activation='relu',subsample=(2,2))) 

    model.add(Flatten())
    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(lr=learning_rate),loss='mse')
  
    return model


'''
Data to be used for learning and validation
'''
data_training_groups = ['data_track_2_right_1', 'data_2']
data_validation_groups = ['data_5', 'data_4']

'''
Parameters for the preprocessing and training phases
'''
epochs = 10
batch_size = 256  
learning_rate = 0.001
dropout = 0.3
# Correction applied when using the left and right images
CORRECTION = .3

# Load the dataframes 
train_samples = du.loadDataFrames(data_training_groups)
validation_samples = du.loadDataFrames(data_validation_groups)


# Data augmentation. From left/right images
train_samples = du.dataFrameWithLeftRight(train_samples, correction=CORRECTION)
validation_samples = du.dataFrameWithLeftRight(validation_samples, correction=CORRECTION)


train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)


samples_per_epoch = len(train_samples)*2
                       
model = getModel()
model.summary()

print ("Total samples , including augmentation " , samples_per_epoch)

history_object = model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, 
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=epochs, verbose=1)
               

model.save('model.h5')


'''
Uncomment to print training/validation loss per epoch
'''
'''
plt.figure(figsize=(12,5))
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''
