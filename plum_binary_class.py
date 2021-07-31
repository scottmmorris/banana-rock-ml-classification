import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, Lambda
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from datetime import date

learning_rate = 0.1
min_learning_rate = 0.00001
learning_rate_reduction_factor = 0.5
patience = 3

verbose = 1

image_size = (100, 100)
input_shape = (100, 100, 3)

use_label_file = False
label_file = 'labels.txt'
base_dir = 'd:/SRFRobotics'
#test_dir = os.path.join(base_dir, 'Test')
#train_dir = os.path.join(base_dir, 'Training')
test_dir = os.path.join(base_dir, 'Test_Banana')
train_dir = os.path.join(base_dir, 'Train_Banana')
output_dir = 'd:/SRFRobotics'  # root folder in which to save the the output files; the files will be under output_files/model_name

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if use_label_file:
    with open(label_file, "r") as f:
        labels = [x.strip() for x in f.readlines()]
else:
    labels = os.listdir(train_dir)

num_classes = len(labels)

train_datagen = ImageDataGenerator(horizontal_flip = True)

test_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_directory(train_dir, target_size=image_size, class_mode='categorical', batch_size=50, shuffle=True, subset='training', classes=labels)

test_gen = test_datagen.flow_from_directory(test_dir, target_size=image_size, class_mode='categorical', batch_size=50, shuffle=False, subset=None, classes=labels)

img_input = Input(shape=input_shape, name='data')
x = Conv2D(16, (5, 5), strides=(1, 1), padding='same', name='conv1')(img_input)
x = Activation('relu', name='conv1_relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool1')(x)
x = Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv2')(x)
x = Activation('relu', name='conv2_relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool2')(x)
x = Conv2D(64, (5, 5), strides=(1, 1), padding='same', name='conv3')(x)
x = Activation('relu', name='conv3_relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool3')(x)
x = Conv2D(128, (5, 5), strides=(1, 1), padding='same', name='conv4')(x)
x = Activation('relu', name='conv4_relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool4')(x)
x = Flatten()(x)
x = Dense(1024, activation='relu', name='fcl1')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu', name='fcl2')(x)
x = Dropout(0.2)(x)
out = Dense(num_classes, activation='softmax', name='predictions')(x)
rez = Model(inputs=img_input, outputs=out)

# Compiling the CNN
rez.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(train_gen)
print(test_gen)

history = rez.fit_generator(train_gen, epochs=200, validation_data = test_gen, verbose=1)

save_model_filename = 'bananarock_classify_model_final'
today = date.today()
date_str = today.strftime("%d%m%y")
save_model_filename = '_'.join([save_model_filename,date_str])
save_model_filename = save_model_filename +'.h5'
rez.save(save_model_filename)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()