import tensorflow as tf
import os
import numpy as np
import json
from keras.models import model_from_json
from PIL import Image


# 1. Pull Model structure from JSON file called 'ageGenderModel.json':
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# 2. Pull Model weights from HDF5 file called 'agrGenderModel.h5':
loaded_model.load_weights("model.h5")
print("Loaded model from disk") # ok, we've got the per-trained model! yei!

# 3. We have to compile and load the model first:
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 4. Data preprocessing:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "unrecognized")

npImages = []  # an numpy array with all numpy images
imgNames = []  # a list with all images titles (just for print)
images = []  # a list of all the images as 'png' or 'jpg' files

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            size = (48, 48)
            img = Image.open(path)
            finalImg = img.resize(size, 3)
            imgArray = np.array(finalImg, "uint8")  # numpy array with values in range [0,255]
            imgArray = imgArray.astype('float32')
            imgArray /= 255.0  # rescale the pixel to [0,1] range (normalize the values)
            npImages.append(imgArray)
            imgNames.append(file.title())
            images.append(img)

npImages = np.stack(npImages)  # predict func only work on numpy arrays

# 5. Making the prediction and print the results:
index = 0

for img in npImages:
    pred = loaded_model.predict(np.array([img]))
    genderLabels = ['Male', 'Female']
    age = int(np.round(pred[1][0]))
    gender = int(np.round(pred[0][0]))
    print("Predicted Age: ", str(age), ", Predicted Gender: ", genderLabels[gender], ", File name: ", imgNames[index])
    imgNames[index] = str(age) + '_' + str(gender) + '.jpg'  # creating new image title to be the age&gender prediction
    index += 1

# 6. Saving the original images with new titles as the prediction in 'finalResults' file
# The title will be in the form: 'age_gender.jpg' or 'age_gender.png' (age and gender are integer values).
# We are doing so because we want to save the predictions in order to display them in the website.
index = 0
image_dir = os.path.join(BASE_DIR, "finalResults")
for img in images:
    name = imgNames[index]
    img.save(image_dir + '/' + name)
    index += 1

print("Images with age and gender estimation has been saved to 'finalResult' file!")







