from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

from src.data import data_loader
from src.visualization.display import show_images_with_data
from src.visualization.roc_curve import show_roc_curve
from src.model.beta_vae_agent import BetaVaeAgent
from configs import config


def get_cosine_similarity(data1, data2):
    scores = []
    for image1, image2 in zip(data1, data2):
        similarity = cosine_similarity(image1.reshape(1, -1), image2.reshape(1, -1))[0][0]
        scores.append(similarity)
    return scores


def show_score_plot(score_data):
    x_bins = np.linspace(min(scores), max(scores), 10)
    for idx, category_scores in enumerate(score_data):
        y, _ = np.histogram(category_scores, bins=x_bins)
        plt.plot(x_bins[:-1], y, label=config.parameter['class_label'][idx])

    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Line plot of Predicted Scores')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    agent = BetaVaeAgent('231008')
    agent.load_model()
    vae = agent.beta_vae

    x_train, y_train, x_test, y_test = data_loader.load_data()
    test_ds = data_loader.tensor_slices(x_test)

    gen_image = vae.predict(test_ds)[0]
    scores = get_cosine_similarity(x_test, gen_image)

    image_data = [[] for _ in range(5)]
    score_data = [[] for _ in range(5)]
    for image, label, score in zip(x_test, y_test, scores):
        image_data[label].append(image)
        score_data[label].append(score)

    class_names = config.parameter['class_label']
    for i in range(len(class_names)):
        show_images_with_data(class_names[i], image_data[i], score_data[i])

    show_roc_curve(y_test, scores)
    show_score_plot(score_data)
