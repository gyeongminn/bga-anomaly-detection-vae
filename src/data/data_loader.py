import numpy as np
from keras.utils import image_dataset_from_directory
from keras.layers import Rescaling
from tensorflow import data

from configs import config


def load_data_gen():
    train_gen = image_dataset_from_directory(
        config.parameter['base_dir'] + '/data/train',
        labels='inferred',
        label_mode='int',
        class_names=['good'],
        color_mode='grayscale',
        batch_size=config.parameter['batch_size'],
        image_size=config.parameter['image_size'],
        seed=config.parameter['random_seed'],
        shuffle=True
    )

    test_gen = image_dataset_from_directory(
        config.parameter['base_dir'] + '/data/test',
        labels='inferred',
        label_mode='int',
        class_names=config.parameter['class_label'],
        color_mode='grayscale',
        batch_size=config.parameter['batch_size'],
        image_size=config.parameter['image_size'],
        seed=config.parameter['random_seed'],
        shuffle=True
    )

    return train_gen, test_gen


def convert_numpy_arr(data_gen):
    x, y = [], []
    for images, labels in data_gen:
        x.extend(images)
        y.extend(labels)

    x = np.array(x)
    y = np.array(y)
    x = x.astype('float32') / 255.

    return x, y


def load_data():
    train_gen, test_gen = load_data_gen()
    x_train, y_train = convert_numpy_arr(train_gen)
    x_test, y_test = convert_numpy_arr(test_gen)

    return x_train, y_train, x_test, y_test


def tensor_slices(x_data):
    batch_size = config.parameter['batch_size']
    return data.Dataset.from_tensor_slices(x_data).batch(batch_size)
