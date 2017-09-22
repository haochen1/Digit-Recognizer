#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 01:32:51 2017

@author: macmac
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#%%
# read training data from CSV file
df=pd.read_csv('train.csv')
print('df size({0[0]},{0[1]})'.format(df.shape))
print(df.head())

images = df.iloc[:,1:].values
images = images.astype(np.float)
# normalization, convert from [0:255] => [0.0:1.0]
images = images/255

print('images size({0[0]},{0[1]})'.format(images.shape))

image_number = images.shape[1]
print ('image_number => {0}'.format(image_number))

# in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_number)).astype(np.int)

print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))

#%%
# display image
def display(img):
    
    # (784) => (28,28)
    one_image = img.reshape(image_width,image_height)
    
    plt.axis('off')
    plt.imshow(one_image, cmap='gray_r')

# image number to output
IMAGE_TO_DISPLAY = 10  
# output image   
display(images[IMAGE_TO_DISPLAY])

#%%
# get labels 
labels = df[['label']].values

print('labels size({0})'.format(len(labels)))
print ('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))

#%%
# count different digits
labels_count = np.unique(labels).shape[0]

print('labels_count => {0}'.format(labels_count))

# convert class labels from scalars to one-hot vectors
def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """  
    # Create a tf.constant equal to C (depth), name it 'C'. 
    C = tf.constant(C, name = "C")   
    # Use tf.one_hot, be careful with the axis 
    one_hot_matrix = tf.one_hot(labels, C,axis=1) #axis=0 one hot vector is a line, axis=1 one hot vector is a colomn    
    # Create the session 
    sess = tf.Session()
    # Run the session
    one_hot = sess.run(one_hot_matrix, feed_dict = {})
    # Close the session 
    sess.close()
    return one_hot

labels_one_hot = one_hot_matrix(labels.ravel(), labels_count)
labels_one_hot = labels_one_hot.astype(np.int)

print('labels_one_hot({0[0]},{0[1]})'.format(labels_one_hot.shape))
print ('labels[{0}] => {1} => {2}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY],labels_one_hot[IMAGE_TO_DISPLAY]))


#%%
# split data into training & dev

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

dev_images = images[:VALIDATION_SIZE]
dev_labels = labels_one_hot[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels_one_hot[VALIDATION_SIZE:]


print('train_images({0[0]},{0[1]})'.format(train_images.shape))
print('dev_images({0[0]},{0[1]})'.format(dev_images.shape))

#%%
# settings
LEARNING_RATE = 1e-4
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 10000       
    
DROPOUT = 0.5
BATCH_SIZE = 64