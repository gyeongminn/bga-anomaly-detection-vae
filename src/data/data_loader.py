import numpy as np
from keras.utils import image_dataset_from_directory
from tensorflow import data

from configs import config


DATA_FOLDER_PATH = config.parameter["base_dir"] + "/data/"

def convert_numpy_arr(data_gen):
    x, y = [], []
    for images, labels in data_gen:
        x.extend(images)
        y.extend(labels)

    x = np.array(x)
    y = np.array(y)
    x = x.astype("float32") / 255.0

    return x, y


def load_train_data():
    train_gen = image_dataset_from_directory(
        DATA_FOLDER_PATH + "train",
        labels="inferred",
        label_mode="int",
        class_names=["good"],
        color_mode="grayscale",
        batch_size=config.parameter["batch_size"],
        image_size=config.parameter["image_size"],
        seed=config.parameter["random_seed"],
        shuffle=True,
    )
    x_train, y_train = convert_numpy_arr(train_gen)

    return x_train, y_train

def load_gen_vae_data():
    train_gen = image_dataset_from_directory(
        DATA_FOLDER_PATH + "gen_vae",
        labels="inferred",
        label_mode="int",
        class_names=["gen"],
        color_mode="grayscale",
        batch_size=config.parameter["batch_size"],
        image_size=config.parameter["image_size"],
        seed=config.parameter["random_seed"],
        shuffle=True,
    )
    x_train, y_train = convert_numpy_arr(train_gen)

    return x_train, y_train


def load_test_data():
    test_gen = image_dataset_from_directory(
        DATA_FOLDER_PATH + "test",
        labels="inferred",
        label_mode="int",
        class_names=config.parameter["class_names"],
        color_mode="grayscale",
        batch_size=config.parameter["batch_size"],
        image_size=config.parameter["image_size"],
        seed=config.parameter["random_seed"],
        shuffle=True,
    )
    x_test, y_test = convert_numpy_arr(test_gen)

    return x_test, y_test


def tensor_slices(x_data):
    batch_size = config.parameter["batch_size"]
    return data.Dataset.from_tensor_slices(x_data).batch(batch_size)


def save_threshold(threshold):

    with open(DATA_FOLDER_PATH + "threshold.txt", "w") as f:
        f.write(str(threshold))


def load_threshold():
    with open(DATA_FOLDER_PATH + "threshold.txt", "r") as f:
        return float(f.read())
