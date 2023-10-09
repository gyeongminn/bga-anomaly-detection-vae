import matplotlib.pyplot as plt
import numpy as np

from configs import config


def show_history(train_loss, kl_loss, recon_loss):
    plt.title("Training history")
    plt.plot(train_loss, label="Training Loss")
    plt.plot(kl_loss, label="KL Loss")
    plt.plot(recon_loss, label="Recon Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_score_plot(score_data, min_score, max_score):
    x_bins = np.linspace(min_score, max_score, 16)
    for idx, category_scores in enumerate(score_data):
        y, _ = np.histogram(category_scores, bins=x_bins)
        plt.plot(x_bins[:-1], y, label=config.parameter["class_label"][idx])

    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.title("Line plot of Predicted Scores")
    plt.legend(loc="upper right")
    plt.show()
