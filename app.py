import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np

def process_image(filepath):    
    width = 512
    height = 512
    image = cv2.imread(filepath)
    imageRescale = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    imageRescale = imageRescale.astype(np.float64)
    imageRescale /= 255
    return imageRescale

def get_file_categories():
    train_label_mapping = {}
    for number, folder in enumerate(os.listdir(os.path.join(os.getcwd(), 'training'))):
        train_label_mapping[number] = folder
    return train_label_mapping

def get_file():
    train_image = []
    train_filename = []
    for number, folder in enumerate(os.listdir(os.path.join(os.getcwd(), 'training'))):
        for image_name in os.listdir(os.path.join(os.getcwd(), 'training', folder)):
            # append folder name as label
            train_filename.append(number)

            # append image to list as data
            image = process_image(os.path.join(os.getcwd(), 'training', folder, image_name))
            train_image.append(image)
    return (train_image, train_filename)

        
def create_model(train_image, train_filename):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=(512, 512, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(3))

    model.summary()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    checkpoint_path = "training/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    history = model.fit(train_image, train_filename, epochs=300, callbacks=[checkpoint_callback])


# train_image, train_filename = get_file()
# train_image = np.array(train_image)
# print(train_image.shape)

# print(train_filename, train_image)
# create_model(train_image, train_filename)