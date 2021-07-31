import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, Lambda
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from datetime import date
from tensorflow.keras.models import load_model

base_dir_path = 'd:/SRFRobotics'
# train_dir_path = os.path.join(base_dir_path,'Training')
train_dir_path = os.path.join(base_dir_path,'Train_Banana')
MNet_InputSize = (100,100)


AllClassNames = os.listdir(train_dir_path)
num_of_classes = len(AllClassNames)
DictOfClasses = {i : AllClassNames[i] for i in range(0, len(AllClassNames))}
    
# Model Prediction on test Images.
ImagePath = 'd:/SRFRobotics/banana5.jpg'
# path_trained_model = os.path.abspath(fruit_classify_model_301220.h5)
trainedModel = load_model('d:/SRFRobotics/bananarock_classify_model_final_280121.h5')


x = image.load_img(ImagePath, target_size=MNet_InputSize)
x = image.img_to_array(x)

#for Display Only
plt.imshow((x * 255).astype(np.uint8))

x = np.expand_dims(x,axis=0)
x = preprocess_input(x)
prediction_class = np.argmax(trainedModel.predict(x,batch_size=1), axis=1)
#prediction_probs = trainedModel.predict_proba(x,batch_size=1)

#print(prediction_probs)
for key, value in DictOfClasses.items():
    if prediction_class == key:
        class_value = value
print(class_value)
