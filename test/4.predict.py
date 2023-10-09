from src.data import data_loader
from src.visualization.image import show_images_with_data
from src.model.parallel_vaes import ParallelVAEs
from src.visualization.image import show_image

if __name__ == "__main__":
    pvae = ParallelVAEs("231008")
    pvae.load_model()

    x_test, y_test = data_loader.load_test_data()

    gen_image_shallow, gen_image_deep = pvae.predict(x_test)

    show_image(x_test[0], "Original image")
    show_image(gen_image_deep[0], "Generated image")

    show_images_with_data("shallow", x_test, y_test)
    show_images_with_data("deep", gen_image_deep, y_test)
