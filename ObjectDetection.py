# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 15:02:35 2021

@author: soheil
"""
import numpy as np
import random

# 1  Generating the inputs ###################################################
n = 70
X_train = np.zeros((n+1,6,10)) # Pre-allocate matrix
y_train = np.zeros(n+1)
for i in range(0,n,2):
    X_train[i,:,:] = np.random.uniform(-0.05,0.05, size=(6, 10))
    y_train[i] = 0
    m=i+1;
    X_train[m,:,:] = np.random.uniform(-0.08,0.08, size=(6, 10))
    y_train[m] = 1
    
    
n = 30
X_test = np.zeros((n+1,6,10)) # Pre-allocate matrix
y_test = np.zeros(n+1)
for i in range(0,n,2):
    X_test[i,:,:] = np.random.uniform(-0.05,0.05, size=(6, 10))
    y_test[i] = 0
    m=i+1;
    X_test[m,:,:] = np.random.uniform(-0.08,0.08, size=(6, 10))
    y_test[m] = 1

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 2 Visualize the First Six Training Images ###################################
# plot first six training images
fig = plt.figure(figsize=(20,20))
for i in range(6):
    ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(str(y_train[i]))
    
# 3 View an Image in More Detail  ###########################################  
def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
visualize_input(X_train[0], ax)   

# 4 Rescale the Images by Dividing Every Pixel in Every Image by 255########## 
# rescale [0,255] --> [0,1]
# X_train = X_train.astype('float32')/255
# X_test = X_test.astype('float32')/255 

# 5 Encode Categorical Integer Labels Using a One-Hot Scheme##################
from tensorflow.keras import utils

#print first ten (integer-valued) training labels
print('Integer-valued labels:')
print(y_train[:2])

# one-hot encode the labels
y_train = utils.to_categorical(y_train, 2)
y_test  =  utils.to_categorical(y_test, 2)

# print first ten (one-hot) training labels
print('One-hot labels:')
print(y_train[:2])

# 6 Define the Model Architecture ############################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,MaxPool2D,Conv2D
X_train = X_train.reshape(-1,6,10,1)
X_test = X_test.reshape(-1,6,10,1)

# define the model
model = Sequential()
#model.add(Flatten(input_shape=X_train.shape[1:]))
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(2, activation='softmax'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
#model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10)
# summarize the model
model.summary()

# 7 Compile the Model ########################################################
# compile the model


# 8 Calculate the Classification Accuracy on the Test Set (Before Training)##
# evaluate test accuracy
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)

# 9 Train the Model  ########################################################
from tensorflow.keras.callbacks import ModelCheckpoint   

# train the model
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', 
                               verbose=1, save_best_only=True)
# hist = model.fit(X_train, y_train, batch_size=128, epochs=10,
#           validation_split=0.2, callbacks=[checkpointer],
#           verbose=1, shuffle=True)


# 10 Calculate the Classification Accuracy on the Test Set ##################
# evaluate test accuracy
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)




