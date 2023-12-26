from sklearn.metrics.pairwise import cosine_similarity

from src.data import data_loader
from src.model.vae_agent import VaeAgent
from src.visualization.image import show_images_with_data

from time import time

if __name__ == "__main__":
    vae = VaeAgent()
    vae.load_model()

    x_test, y_test = data_loader.load_test_data()
    y_test_bin = [1 if x == 0 else 0 for x in y_test]

    start = time()
    gen_data = vae.predict(x_test)

    scores = []
    for x, y in zip(x_test, gen_data):
        similarity = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]
        scores.append(similarity)

    optimal_threshold = data_loader.load_threshold()

    result_image = []
    result_score = []
    for image, score in zip(x_test, scores):
        result_image.append(image)
        result_score.append(score)
        label = "Good" if optimal_threshold < score else "Defective"

    end = time()
    print(f"inspection time ({len(x_test)} image): {(end - start):.3f}s")
    print(f"inspection time (1 image): {((end - start) / len(x_test) * 1000):.3f}ms")

    show_images_with_data("Result", result_image, label)
