

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.applications.vgg19 import VGG19
from glob import glob
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

image_size=[224,224]
train_data="/content/drive/MyDrive/Dataset/Train"
test_data="/content/drive/MyDrive/Dataset/Test"

vgg19=VGG19(input_shape=image_size+[3], weights="imagenet", include_top= False)


#storing each layer in an address and setting training off.
for layer in vgg19.layers:
  layer.trainable=False

folders=glob("/content/drive/MyDrive/Dataset/Train/*")

x=Flatten()(vgg19.output)

prediction= Dense(len(folders), activation='softmax')(x)

model=Model(inputs=vgg19.input, outputs=prediction)

model.summary()

#to tell the model what loss funtion and optimization method to use, we use model.compile
model.compile(loss="categorical_crossentropy",optimizer="adam", metrics=["accuracy"])

#to read the data we use image data generator to import the data from the dataset
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory("/content/drive/MyDrive/Dataset/Train",target_size=(224,224),batch_size=32,class_mode="categorical")

test_set=test_datagen.flow_from_directory("/content/drive/MyDrive/Dataset/Test",target_size=(224,224),batch_size=32,class_mode="categorical")

#fit the model for training and validation
r=model.fit(training_set,validation_data=test_set, epochs=2, steps_per_epoch=len(training_set), validation_steps=len(test_set))

#losses
plt.plot(r.history['loss'],"r",label='train loss')
plt.plot(r.history['val_loss'],"b",label='train val_loss')
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

#accuracies
plt.plot(r.history['accuracy'],"orange",label='train accuracy')
plt.plot(r.history['val_accuracy'],"g",label='train val_accuracy')
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
plt.savefig('AccVal_acc') 

model.save('model_vgg19.h5')
