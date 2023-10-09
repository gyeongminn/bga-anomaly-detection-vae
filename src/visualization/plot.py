from sklearn.metrics import auc
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
    plt.figure(figsize=(8, 6))
    for idx, category_scores in enumerate(score_data):
        sns.kdeplot(
            category_scores, fill=True, label=config.parameter["class_names"][idx]
        )
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.title("Density plot of predicted scores")
    plt.legend(loc="upper left")
    plt.show()


def show_roc_curve(fpr, tpr):
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
