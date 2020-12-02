import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import sys
import pathlib
import json

batch_size = 32
img_height = 180
img_width = 180
num_classes = 5

class_names = None


def downloadDataset():
  dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
  data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
  data_dir = pathlib.Path(data_dir)
  return data_dir


def settings(data_dir):
  global class_names

  # Training settings
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, # Using 20% for validation
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  # Validatino settings
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  # Class names in alphabetical order
  class_names = train_ds.class_names
  saveClassNames(class_names)

  # Configuring the dataset for performance
  AUTOTUNE = tf.data.experimental.AUTOTUNE

  train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  return train_ds, val_ds


def augmentation():
  # Model based on Augmentation
  return keras.Sequential(
    [
      layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                  input_shape=(img_height, 
                                                                img_width,
                                                                3)),
      layers.experimental.preprocessing.RandomRotation(0.1),
      layers.experimental.preprocessing.RandomZoom(0.1),
    ]
  )


def dropout(data_augmentation):
  return Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
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
  epochs = 15
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )

  return model


def check(model):
  # sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
  # sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

  path = "./photos/paardenbloem.jpeg"

  img = keras.preprocessing.image.load_img(
      path, target_size=(img_height, img_width)
  )
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  return predictions, score


def saveModel(model):
  model.save('./saves/model')


def loadModel():
  return keras.models.load_model('./saves/model') 


def saveClassNames():
  with open('class_names.json', 'w') as file:
    json.dump(class_names, file)


def loadClassNames():
  with open('class_names.json') as file:
      names = json.load(file)

  return names


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1].lower() == "new":
    train_ds, val_ds = settings(downloadDataset())
    model = training(compileModel(dropout(augmentation())), train_ds, val_ds)
    saveModel(model)
  else:
    model = loadModel()
    class_names = loadClassNames()
    
  predictions, score = check(model)

  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )