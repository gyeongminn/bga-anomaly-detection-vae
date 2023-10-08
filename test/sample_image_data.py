from src.data import data_loader
from src.visualization.image import show_image
from configs import config

if __name__ == "__main__":
    train_ds, test_ds = data_loader.load_data_gen()

    class_names = config.parameter["class_label"]
    images, labels = next(iter(test_ds))

    label = class_names[labels[0]]
    show_image(images[0], label)
