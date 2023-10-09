from tensorflow.python.ops.numpy_ops import np_config
from sklearn.metrics.pairwise import cosine_similarity

from src.data import data_loader
from src.visualization.image import show_images_with_data
from src.visualization.roc_curve import show_roc_curve
from src.visualization.plots import show_score_plot
from src.model.parallel_vaes import ParallelVAEs
from configs import config

show_image = True


def cosine_score(data1, data2):
    np_config.enable_numpy_behavior()
    scores = []
    for x, y in zip(data1, data2):
        similarity = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]
        scores.append(similarity)
    return scores


if __name__ == "__main__":
    pvae = ParallelVAEs("231008")
    pvae.load_model()

    x_test, y_test = data_loader.load_test_data()

    for gen_data in pvae.predict(x_test):
        scores = cosine_score(x_test, gen_data)
        image_data = [[] for _ in range(5)]
        score_data = [[] for _ in range(5)]
        for image, label, score in zip(x_test, y_test, scores):
            image_data[label].append(image)
            score_data[label].append(score)

        class_names = config.parameter["class_label"]

        show_roc_curve(y_test, scores)
        show_score_plot(score_data, min(scores), max(scores))

        if show_image:
            for i in range(len(class_names)):
                show_images_with_data(
                    class_names[i], image_data[i], score_data[i], is_float=True
                )
