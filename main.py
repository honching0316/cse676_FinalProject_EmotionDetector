# -*- coding: utf-8 -*-
"""
Created on Sun May 13 20:15:26 2022

@author: Hon Ching Li, ___, ____
"""


#%%
'''Import libraries
    Need following commands if never installed - 
    pip install tensorflow
    pip install keras
    Just ignore the GPU warning, could deal with it tho
    reason why we using block - could avoid building the model over and over
    again everytime we make any minor changes
'''
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.preprocessing import image

# to show image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




#%%
'''Data Preprocessing
    Training set needs image augmentation/transformation and feature scaling
    augmentation - to avoid overfitting
    Testing set - only use featrue scaling, never apply augmentation!
    Supervise learning, so model could get the accuracy easily
    Each image class/label is represent by it's own folder name
    eg. all images within 'angry' folder gonna be class of 'angry'
'''


# Training set

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
# batch size = amount of images pass into the NN at single time, general = 32
training_set = train_datagen.flow_from_directory(
        './images/train',
        target_size=(48, 48),
        batch_size=64,
        class_mode='categorical',
        color_mode="grayscale")


# Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        './images/validation',
        target_size=(48, 48),
        batch_size=64,
        class_mode='categorical',
        color_mode="grayscale")


#%%
'''Building the CNN model'''


# initializing
# it's cnn model
model = tf.keras.models.Sequential()


# add a convolutional layer - 1. convolution, 2. pooling
# 1. convolution, our images are in black white style, no color channel
# that's why the input size is 56 56 1(channel)
model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = \
                               'relu', input_shape = [48, 48, 1]))

# 2. pooling
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides=2))


# adding 2nd convolutional layer
# still dropout is to prevent overfitting
model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, 
                                 activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides=2))

# adding 3rd convolutional layer
model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, 
                                 activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides=2))


# Flattening layer - flattening the 'pooled feature maps' into 1D array
# for forward neural network
model.add(tf.keras.layers.Flatten())

# add 1 fully connected hidden layer
model.add(tf.keras.layers.Dense(units=128, activation='relu'))



# add a output layer
# we have multiclass (5 classes) classification, so units = 5
# for multiple classification, usually use softmax
model.add(tf.keras.layers.Dense(units=5, activation = 'softmax'))


#%%
'''Train the CNN model'''


'''compile the cnn'''
# 'adam' = Stochastic Gradient Descent
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
            metrics = ['accuracy'])



#%%
'''Training cnn, meanwhile the testset going to be evaluate at the same time
'''

history = model.fit(x=training_set, validation_data = test_set, epochs = 30)



#%%
''' Show the accuracy Plot it out
'''
print(model.metrics_names)

plt.title('CNN Model Accuracy')

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.title('CNN Model Accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['training set', 'testing set'], loc='upper left')

plt.subplot(1, 2, 2)
plt.title('CNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.grid(True)
plt.legend(loc='upper right')
# plt.show()


#%%

'''predict 1 single image's emotion at a time '''

'''
The following aren't train set, these aren't be train into model, so
it's legit to use it to test our model's predicted results
./images/validation/angry/157.jpg
./images/validation/angry/65.jpg
./images/validation/sad/70.jpg
'''

# img = mpimg.imread('./images/validation/sad/70.jpg')
# plt.imshow(img)

test_image = image.load_img('./images/validation/sad/70.jpg', 
                            target_size=(48, 48), color_mode = "grayscale")
# must into 1D array and normalize for the predict method
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
prediction = model.predict(test_image/255.0)

print(training_set.class_indices)

''' discover that each index in prediction[0] has 1 percentage number
 corresponding to the angry, happy, neutral, sad, surprise
 the highest percentage is the predicted result
 the following print statement will help to visualize
'''
#print(type(prediction))
print(prediction[0])

most_possible = -1;
most_poss_idx = -1;

for i in range(len(prediction[0])):
    if(most_possible<prediction[0][i]):
        most_possible = prediction[0][i]
        most_poss_idx = i

res_lis = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

if most_poss_idx>=0:
    res = res_lis[most_poss_idx]
    print("This face is very likely to be " + '"' + res + '"')
else:
    print("Face invalid")


# if prediction[0][0] > 0.5:
#     prediction = 'dog'
# else:
#     prediction = 'cat'
    
# print(prediction)








