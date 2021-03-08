import tensorflow as tf
import os
import numpy as np
import json
from keras.models import model_from_json
from PIL import Image
import requests
import datetime
import base64


# 1. Pull Model structure from JSON file called 'model.json'.
json_file = open('faceRecognitionModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# 2. Pull Model weights from HDF5 file called 'model.h5'.
loaded_model.load_weights("faceRecognitionModel.h5")
print("Loaded model from disk")  # ok, we've got the per-trained model! yei!

# 3. We have to compile and load the model first.
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 4. Preparing numpy arrays with correct input shape (224,224,3) for face recognition model from
# persons images in order to do predictions.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "testImages")  # the path to the test images file

testImages = []  # a list of all the images to do prediction on
imgNames = []  # a list of all the images titles (just for print and see if correct)
images = []  # a list of all the images as '.jpg' or '.png' files (in order to save if images isn't recognized

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            img = Image.open(path)
            size = (224, 224)
            finalImg = img.resize(size, 3)
            imgArray = np.array(finalImg, "uint8")  # numpy array with values in range [0,255]
            imgArray = imgArray.astype('float32')
            imgArray /= 255.0  # rescale the pixel to [0,1] range (normalize the values)
            testImages.append(imgArray)
            imgNames.append(file.title())
            images.append(img)

# 5. Read 'labels.txt' file and create a list with persons ID's.
labelsFile = open('labels.txt', 'r')
lines = labelsFile.readlines()
labels = []
for line in lines:
    line = line.strip()
    labels.append(line)

# 6. Making the prediction and print the results.
testArray = np.stack(testImages, axis=0)  # predict func only work on numpy arrays
predictionVector = loaded_model.predict(testArray)

# Here I need to check the distribution for each prediction vector:
# For my opinion, we need to find bottom bound that underneath it the person in not in dataset.
# We must put more images for each individual in order to get more accuracy!!!
# Or, alternatively, to improve the quality of the test images.
imgIndex = 0  # variable for myself to see the current predicted image name
unrecognizedImages = []  # an array of all the images we failed in recognition
recognizedImages = []  # an array of all the people ID's we recognized - we might change it to some other structure

for prediction in predictionVector:
    maxMatch = np.amax(prediction)  # the best match for the image
    print(imgNames[imgIndex])
    imgIndex += 1
    if maxMatch < 0.25:
        print("This person is not in the dataset!")
        unrecognizedImages.append(images[imgIndex])
        # Go to Age&Gender estimation...
    else:
        result = np.where(prediction == np.amax(prediction))  # finding the best match from all classes
        index = result[0][0]
        personId = labels[index]
        print("The person in the image is: ", personId, ", The match is: ", maxMatch)
        if personId not in recognizedImages:
            recognizedImages.append(personId)
        # Move person ID to web API...


# 7. Saving the unrecognized images in unrecognized file
unrecognized_dir = os.path.join(BASE_DIR, "unrecognized")
name = 1

for img in unrecognizedImages:
    with open(unrecognized_dir+'/'+str(name)+'.jpg', 'w') as f:
        img.save(f)
    name += 1


# 8. Saving the ID's of the persons we recognized (to be continue...)
date = datetime.datetime.now()
for personId in recognizedImages:
    im = Image.open('./trainImages/' + personId + '/face0.jpg')
    im.save('./trainImages/' + personId + '/face0.png')
    with open('./trainImages/' + personId + '/face0.png', 'rb') as binary_file:
        binary_file_data = binary_file.read()
        base64_encoded_data = base64.b64encode(binary_file_data)
        base64_message = base64_encoded_data.decode('utf-8')

    response = requests.post('https://shirtal.nbeno.com/API/Persons/GetPersonByIdNumber?IdNumber=' + personId)
    personJson = json.loads(response.text)
    response2 = requests.post('https://shirtal.nbeno.com/API/Operations/AddOrUpdateOperation',
                              data={'Id': '0', 'ExitEntrance': 'Entrance', 'PersonId': personJson["Id"],
                                    'TempPersonId': 'null',
                                    'TimeStamp': date, 'BuildingId': '14', 'EstimatedAge': 'null',
                                    'EstimatedGender': 'null',
                                    'Image': base64_message})
    #print(response2.status_code)

with open('recognized.txt', 'w') as f:
    for item in recognizedImages:
        f.write('%s\n' % item)





