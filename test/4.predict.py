from src.data import data_loader
from src.model.vae_agent import VaeAgent
from src.visualization.image import show_image, show_images_with_data

if __name__ == "__main__":
    vae = VaeAgent()
    vae.load_model()

    x_test, y_test = data_loader.load_test_data()

    gen_image = vae.predict(x_test)

    show_image(x_test[0], "Original image")
    show_image(gen_image[0], "Generated image")

    labels = [vae.class_names[i] for i in y_test]
    show_images_with_data("Original images", x_test, labels)
    show_images_with_data("Generated images", gen_image, labels)
