import tensorflow as tf
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D
from tensorflow.keras.layers import Input, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "UTKFace dataset/UTKFace")

# 1. Data preprocessing:
ages = []   # a list of all ages from training images (ages in range [0-116])
genders = []  # a list of all genders from training images ( Male - 0 , Female - 1 )
images = []  # a list of all training images

# Creating images, age and gender lists from the dataset:
files = os.listdir(data_dir)  # the titles of all the files (images) in dataset

# The title of each image file in 'UTKFace' dataset is in the form:
# 'int-age_boolean-gender_int-ethnic_person-id.jpg'.
# The model gets an input in the shape: (48,48,3). Hence, we need to resize the images
# and change from 'BGR' (the default of open-cv) to 'RGB' ( 3 - the input for the model )
for file in files:
    age = int(file.split('_')[0])
    gender = int(file.split('_')[1])
    total = data_dir+'/'+file
    image = cv2.imread(total)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    images.append(image)
    ages.append(age)
    genders.append(gender)

# Creating the labels for the training set:
# the labels shape will be:
# [[[age(1)],[gender(1)]],
# [[age(2)],[gender(2)]], ……………….[[age(n)],[gender(n)]]]
labels = []
i = 0
while i < len(ages):
    label = []
    label.append([ages[i]])
    label.append([genders[i]])
    labels.append(label)
    i += 1

# Converting the images and labels lists to numpy arrays:
images_f = np.array(images)
labels_f = np.array(labels)

# Normalize the images from range [0-255] to [0-1]:
images_f_2 = images_f/255.0

# Creating the training and testing data split (we will use a 25% test split):
X_train, X_test, Y_train, Y_test = train_test_split(images_f_2, labels_f, test_size=0.2)

# Transform the Y_train and Y_test to a form that Y_train[0] denotes the gender labels vector,
# and Y_train[1] denotes the age labels vector:
Y_train_2 = [Y_train[:, 1], Y_train[:, 0]]
Y_test_2 = [Y_test[:, 1], Y_test[:, 0]]

# 2. Building the model:

# Convolution function gets the input shape and the number of filters and create conv2D layer.
def Convolution(input_tensor, filters):
    x = Conv2D(filters=filters,
               kernel_size=(3, 3),
               padding='same',
               strides=(1, 1),
               kernel_regularizer=l2(0.001))(input_tensor)
    x = Dropout(0.1)(x)
    x = Activation('relu')(x)
    return x


def model(input_shape):
    inputs = Input((input_shape))
    conv_1 = Convolution(inputs, 32)
    maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2 = Convolution(maxp_1, 64)
    maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    conv_3 = Convolution(maxp_2, 128)
    maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    conv_4 = Convolution(maxp_3, 256)
    maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)
    flatten = Flatten()(maxp_4)
    dense_1 = Dense(64, activation='relu')(flatten)
    dense_2 = Dense(64, activation='relu')(flatten)
    drop_1 = Dropout(0.2)(dense_1)
    drop_2 = Dropout(0.2)(dense_2)
    output_1 = Dense(1, activation="sigmoid", name='sex_out')(drop_1)
    output_2 = Dense(1, activation="relu", name='age_out')(drop_2)
    model = Model(inputs=[inputs], outputs=[output_1, output_2])
    model.compile(loss=["binary_crossentropy", "mae"], optimizer="Adam",
                  metrics=["accuracy"])
    return model


# Creating the Model using the model() function (model input shape: (48,48,3):
Model = model((48, 48, 3))
Model.summary()

# **Gender prediction is a classification problem:
# We will use sigmoid as output activation for gender prediction.
# We will use ‘binary cross-entropy’ as the loss function for gender.
# **Age prediction is a regression problem.
# We will use ReLU as the activation function for age prediction.
# We will use ‘mean absolute error’ as the loss function for the age prediction.

# 3. Saving in 'Age_sex_detection.h5' file the best features for the model.
fle_s = 'Age_sex_detection.h5'

checkpointer = ModelCheckpoint(fle_s,
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=False,
                               mode='auto',
                               save_freq='epoch')

Early_stop = tf.keras.callbacks.EarlyStopping(patience=75,
                                              monitor='val_loss',
                                              restore_best_weights=True),
callback_list = [checkpointer, *Early_stop]

# 4. Training the model:
History = Model.fit(X_train,
                    Y_train_2,
                    batch_size=128,
                    validation_data=(X_test, Y_test_2),
                    epochs=50,
                    callbacks=callback_list)

# 5. Evaluate model accuracy:
Model.evaluate(X_test, Y_test_2)

# 6. Saving the model structure in JSON file and the model weights in HDF5 file:
model_json = Model.to_json()
with open("ageGenderModel.json", "w") as json_file:
    json_file.write(model_json)

Model.save_weights("ageGenderModel.h5")
