import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from src.data import data_loader
from src.model.vae_agent import VaeAgent
from src.visualization.image import show_images_with_data
from src.visualization.plot import show_score_density_plot, show_roc_curve

np_config.enable_numpy_behavior()

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

    end = time()
    print(f"inspection time ({len(x_test)} image): {(end - start):.3f}s")
    print(f"inspection time (1 image): {((end - start) / len(x_test) * 1000):.3f}ms")

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

    fpr, tpr, thresholds = roc_curve(y_test_bin, scores)
    show_roc_curve(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    data_loader.save_threshold(optimal_threshold)

    y_pred = [1 if score >= optimal_threshold else 0 for score in scores]

    accuracy = accuracy_score(y_test_bin, y_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(
        y_test_bin, y_pred
    )
    report = classification_report(
        y_test_bin, y_pred, target_names=["defective", "good"]
    )

    print(f"threshold: {optimal_threshold}")
    print()
    print("Accuracy   : %.3f" % accuracy)
    print("Precision  : %.3f" % precision[0])
    print("Recall     : %.3f" % recall[0])
    print("Specificity: %.3f" % recall[1])
    print("F1-Score   : %.3f" % f_score[0])
    print()
    print(report)
