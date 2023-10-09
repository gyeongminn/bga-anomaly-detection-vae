import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    classification_report,
    roc_curve,
)

from src.data import data_loader
from src.model.vae_agent import VaeAgent

np_config.enable_numpy_behavior()

if __name__ == "__main__":
    vae = VaeAgent()
    vae.load_model()

    x_test, y_test = data_loader.load_test_data()
    y_test_bin = [1 if x == 0 else 0 for x in y_test]

    gen_data = vae.predict(x_test)

    scores = []
    for x, y in zip(x_test, gen_data):
        similarity = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]
        scores.append(similarity)

    fpr, tpr, thresholds = roc_curve(y_test_bin, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"idx: {optimal_idx}\tthreshold: {optimal_threshold}")

    pred = [1 if score >= optimal_threshold else 0 for score in scores]
    print(classification_report(y_test_bin, pred, target_names=["defective", "good"]))
