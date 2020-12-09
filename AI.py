import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import cv2
import time
import os.path


#import tensorflow_io as tfio

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import sys
import pathlib
import json
import os

# Init camera
frameWidth = 1920
frameHeight = 1080
cap = cv2.VideoCapture(0)
cap.set(16, frameWidth)
cap.set(9, frameHeight)

# Output
cropsize = (150, 50)

# Save output
Dir = "./crop/"
run = 0
filecount = next(os.walk(Dir))[2]
run = (len(filecount) - 1)

# AI files
batch_size = 10
img_height = 50
img_width = 150
epochs = 15

num_classes = None
class_names = None


def downloadDataset():
    data_dir = os.getcwd() + "/dataset"
    return data_dir


def creatingDatasetSettings(data_dir, split, type):
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=split,  # Using % for validation
        subset=type,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)


def settings(data_dir):
    global class_names, num_classes

    # Training settings
    train_ds = creatingDatasetSettings(data_dir, 0.3, "training")

    # Validation settings
    val_ds = creatingDatasetSettings(data_dir, 0.3, "validation")

    # Class names in alphabetical order
    class_names = train_ds.class_names
    num_classes = len(class_names)
    saveClassNames(class_names)

    # Configuring the dataset for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # showNinePhotos(train_ds)

    return train_ds, val_ds


def showNinePhotos(train_ds):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()


def createSequential():
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(img_height,
                                                                      img_width,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.2),
        ]
    )

    return Sequential([
        # data_augmentation,
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])


def compileModel(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def training(model, train_ds, val_ds):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return model, history


def showImage(img):
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def check(model, path):
    img = keras.preprocessing.image.load_img(
        path, target_size=(img_height, img_width)
    )

    # showImage(img)

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return predictions, score


def saveModel(model):
    model.save('./saves/model')


def loadModel():
    return keras.models.load_model('./saves/model')


def saveClassNames(classes):
    with open('class_names.json', 'w') as file:
        json.dump(classes, file)


def loadClassNames():
    with open('class_names.json') as file:
        names = json.load(file)

    return names


def visualizeData(history):
    # Visualizing data with Dropout & Augmentation
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# Stack images in one window (mostly for debugging)
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# Function getContours
def getContours(img, orig):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4000:                                                               # Offset for wrong readings
            # print(area)
            cv2.drawContours(imgContour, [cnt], 0, (255, 0, 0), 3)                    # Draw contour
            peri = cv2.arcLength(cnt, False)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, False)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if w >= 100 and h >= 30:                                                  # Offset for contour mismatch
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)     # Show rectangle around contour
                x = x + 230                                                           # Crop only middle part of resistor
                w = w - 460                                                           # " "

                imgCropped = orig[y:y + h, x:x + w]                                   # Crop resistor
                imgCropped = cv2.resize(imgCropped, cropsize, 0, 0, cv2.INTER_AREA)   # Resize crop to 150, 50 pixels

                cv2.imshow("Crop", imgCropped)                                        # Show cropped image
                time.sleep(0.5)                                                       # Sleep for stability
                global run                                                            # Counter for how many samples
                run = run + 1
                print(run)
                cv2.imwrite("./crop/" + str(run) + ".jpg", imgCropped)  # Save cropped image
                global path
                path = "./crop/" + str(run) + ".jpg"
                predictions, score = check(model, path)

                print(
                    "This is a {} resistor with a {:.2f} % confidence."
                        .format(class_names[np.argmax(score)], 100 * np.max(score))
                )


# Main
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "new":
        train_ds, val_ds = settings(downloadDataset())
        model, history = training(compileModel(createSequential()), train_ds, val_ds)
        saveModel(model)
        # visualizeData(history)
    else:
        model = loadModel()
        class_names = loadClassNames()

    # paths = [
    #     "./photos/resistors/220.jpeg",
    #     "./photos/resistors/5.6k.jpeg",
    #     "./photos/resistors/1M.jpeg",
    #     "./photos/resistors/1k.jpeg"
    # ]

    while True:
        success, img = cap.read()  # Open camera
        imgContour = img.copy()  # Copy image
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gray filter
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), cv2.BORDER_REPLICATE)  # Blur filter
        thresh = cv2.threshold(imgBlur, 200, 255, cv2.THRESH_BINARY)[1]  # Threshhold for glare
        thresh = cv2.erode(thresh, None, iterations=5)  # Erode filter
        thresh = cv2.dilate(thresh, None, iterations=5)  # Dilate filter
        imgCanny = cv2.Canny(thresh, 50, 50)  # Canny filter

        getContours(imgCanny, img)  # Call function getContours
        imgStack = stackImages(1, ([img, imgCanny, imgContour]))  # Stack input, cannyfilter and contours
        cv2.imshow("Stack", imgStack)  # Show stack in one window

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Show stack until next image or 'q'
            break

    # for path in paths:
    #     predictions, score = check(model, path)
    #     print()
    #     print(path)
    #     print()
    #
    #     print(
    #         "This image most likely belongs to {} with a {:.2f} percent confidence."
    #             .format(class_names[np.argmax(score)], 100 * np.max(score))
    #     )
    #
    #     print()
    #     print("Predictions:")
    #     print("1M: " + str(round(np.max(score[0]) * 100)) + "%")
    #     print("1k: " + str(round(np.max(score[1]) * 100)) + "%")
    #     print("220: " + str(round(np.max(score[2]) * 100)) + "%")
    #     print("5.6k: " + str(round(np.max(score[3]) * 100)) + "%")

