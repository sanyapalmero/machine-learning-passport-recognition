import os

import cv2
import numpy
from matplotlib import pyplot
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import img_to_array


# Settings setup
IMAGE_MAX_SIZE = 200, 200  # Witdh and heingt in px
TRAIN_PASSPORTS_FOLDER = "passports-training-dataset/"
TEST_PASSPORTS_FOLDER = "passports-test-dataset/"
CLASS_NAMES = ['passport', 'not_passport']


# Temporarily disable tensorflow logs because we don't use GPU now
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def prepare_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, IMAGE_MAX_SIZE)
    image = img_to_array(image)
    return image


def load_image_dataset(path_dir):
    images = []
    labels = []
    for file in os.listdir(path_dir):
        try:
            img = prepare_image(path_dir + file)
        except Exception:
            continue
        images.append(img)

        if "not_passport" in file:
            labels.append(1)
        else:
            labels.append(0)

    return numpy.array(images), numpy.array(labels)


def display_images(images, labels):
    pyplot.figure(figsize=(10, 10))
    for i in range(25):
        pyplot.subplot(5, 5, i+1)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.grid(False)
        pyplot.imshow(images[i], cmap=pyplot.cm.binary)
        pyplot.xlabel(CLASS_NAMES[labels[i]])


def create_model():
    model = Sequential()

    width, height = IMAGE_MAX_SIZE
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2))

    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


def train_model(model, train_images, train_labels, epochs):
    model.fit(train_images, train_labels, epochs=epochs)
    model.save('model.h5')
    return model


def check_model_accuracy(model, test_images, test_labels):
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f'Model accuracy: {accuracy}. Model loss: {loss}')


def recognize(model, images):
    predictions = model.predict(images)
    print(predictions)
    display_images(images, numpy.argmax(predictions, axis=1))
    pyplot.savefig("results.png")


def main():
    print("Prepare train images...")
    train_images, train_labels = load_image_dataset(TRAIN_PASSPORTS_FOLDER)
    train_images = train_images / 255.0
    display_images(train_images, train_labels)
    pyplot.savefig("train_images.png")
    print("Train images was prepared, you can check train_images.png file.")

    print("Prepare test images...")
    test_images, test_labels = load_image_dataset(TEST_PASSPORTS_FOLDER)
    test_images = test_images / 255.0
    display_images(test_images, test_labels)
    pyplot.savefig("test_images.png")
    print("Test images was prepared, you can check test_images.png file.")

    retrain = True
    model_file = "model.h5"
    if os.path.exists(model_file) and not retrain:
        print("Found model.h5 file. Load existed model.")
        model = load_model(model_file)
    else:
        print("Create new model...")
        model = create_model()
        epochs = 10
        print(f"Start model training on {epochs} epochs.")
        model = train_model(model, train_images, train_labels, epochs)
        print("Model training was successfully finished.")

    check_model_accuracy(model, test_images, test_labels)

    print("Start recognizing...")
    recognize(model, test_images)


if __name__ == "__main__":
    main()
