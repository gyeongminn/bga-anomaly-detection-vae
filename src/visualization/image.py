import matplotlib.pyplot as plt


def show_image(image, title=None):
    if title:
        plt.title(title, fontsize=20)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def show_images_with_data(title, images, data, is_float=False):
    row, col = min(len(images) // 5, 10), 5
    fig, axes = plt.subplots(
        nrows=row, ncols=col, figsize=(20, 4 * row), constrained_layout=True
    )
    fig.suptitle(title, fontsize=40)
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            ax.imshow(images[i], cmap="gray")
            if is_float:
                ax.set_title(f"\n{data[i]:.4f}", fontsize=20)
            else:
                ax.set_title(f"\n{data[i]}", fontsize=20)
        ax.axis("off")
    plt.show()
