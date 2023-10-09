from tensorflow.python.ops.numpy_ops import np_config
from sklearn.metrics.pairwise import cosine_similarity

from src.data import data_loader
from src.model.vae_agent import VaeAgent
from src.visualization.image import show_images_with_data
from src.visualization.result import show_roc_curve
from src.visualization.plot import show_score_density_plot

np_config.enable_numpy_behavior()

if __name__ == "__main__":
    vae = VaeAgent()
    vae.load_model()

    x_test, y_test = data_loader.load_test_data()

    gen_data = vae.predict(x_test)

    scores = []
    for x, y in zip(x_test, gen_data):
        similarity = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]
        scores.append(similarity)

    show_roc_curve(y_test, scores)

    class_images = [[] for _ in range(len(vae.class_names))]
    class_scores = [[] for _ in range(len(vae.class_names))]
    for image, label, score in zip(x_test, y_test, scores):
        class_images[label].append(image)
        class_scores[label].append(score)

    show_score_density_plot(class_scores)

    show_image = False
    if show_image:
        for i in range(len(vae.class_names)):
            show_images_with_data(
                vae.class_names[i], class_images[i], class_scores[i], is_float=True
            )
