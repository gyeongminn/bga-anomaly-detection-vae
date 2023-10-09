import matplotlib.pyplot as plt
import seaborn as sns

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


def show_score_density_plot(score_data):
    for idx, category_scores in enumerate(score_data):
        sns.kdeplot(
            category_scores, shade=True, label=config.parameter["class_names"][idx]
        )

    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.title("Density plot of predicted scores")
    plt.legend(loc="upper left")
    plt.show()
