from src.data import data_loader
from src.visualization import display
from configs import config

if __name__ == "__main__":
    train_ds, test_ds = data_loader.load_data_gen()

    class_names = config.parameter['class_label']
    images, labels = next(iter(test_ds))

    label = class_names[labels[0]]
    display.show_image(images[0], label)
