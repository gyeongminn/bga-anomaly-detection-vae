from src.data import data_loader
from src.visualization.image import show_image
from configs import config

if __name__ == "__main__":
    x_train, y_train = data_loader.load_train_data()

    class_label = config.parameter["class_names"]
    label = class_label[y_train[0]]

    show_image(x_train[0], label)
