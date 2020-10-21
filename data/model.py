import os
import numpy as np
import random,shutil
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
#Importing all the reuired modules
#Function to process training and validation sets
def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):
    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)
#Declaring Batch Size    
BS= 32
#Declaring image size
TS=(24,24)
#Importing database images
train_batch= generator('v_data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('v_data/test',shuffle=True, batch_size=BS,target_size=TS)
#Displaying database details
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)
#Declaring a Sequential Model
model = Sequential()
#Conv2D is the layer to convolve the image into multiple images
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(24,24,1)))
#MaxPooling2D is used to max pool the value from the given size matrix
model.add(MaxPooling2D(pool_size=(1,1)))
#Conv2D is the layer to convolve the image into multiple images
model.add(Conv2D(32,(3,3),activation='relu'))
#MaxPooling2D is used to max pool the value from the given size matrix
model.add(MaxPooling2D(pool_size=(1,1)))
#Conv2D is the layer to convolve the image into multiple images
model.add(Conv2D(64, (3, 3), activation='relu'))
#MaxPooling2D is used to max pool the value from the given size matrix
model.add(MaxPooling2D(pool_size=(1,1)))
#Dropout is used to avoid overfitting on the dataset
model.add(Dropout(0.25))
model.add(Flatten())
#Dense is used to make this a fully connected model and is the hidden layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
#Compiliation function to optimise and reduce loss.
#optimizer used is adam ad loss function used is categorical_crossentropy
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#fit_generator is used to fit the data into the model made above
#15 epochs means, it will be trained 15 times
model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)
#Saving the model so as it can be used anywhere without training it again and again
model.save('eye_classifier_cnn.h5', overwrite=True)