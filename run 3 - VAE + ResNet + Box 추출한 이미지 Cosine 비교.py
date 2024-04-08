import cv2
import numpy as np
from time import time

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from tensorflow.keras.applications.resnet50 import preprocess_input

from src.data import data_loader
from src.model.vae_agent import VaeAgent
from src.visualization.image import show_images_with_data, show_image
from src.visualization.plot import show_score_density_plot, show_roc_curve
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model


def create_feature_extractor():
    base_model = ResNet50(weights='imagenet', include_top=False)

    layer2_output = base_model.get_layer('conv3_block4_out').output
    layer3_output = base_model.get_layer('conv4_block6_out').output

    flattened_layer2_output = tf.keras.layers.Flatten()(layer2_output)
    flattened_layer3_output = tf.keras.layers.Flatten()(layer3_output)
    concatenated_output = tf.keras.layers.Concatenate(axis=-1)([flattened_layer2_output, flattened_layer3_output])

    model = Model(inputs=base_model.input, outputs=concatenated_output)

    return model


def extract_features(model, image):
    img_array = preprocess_image(image)
    features = model.predict(img_array)
    return features


def preprocess_image(img_array):
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def apply_mask_for_each_box(image, boxes):
    result_image = np.zeros_like(image)

    for box in boxes:
        x, y, w, h = box
        result_image[y:y+h, x:x]
        result_image[x:x + w, y:y + h] = image[x:x + w, y:y + h]

    return result_image


def get_bounding_boxes(image, size=64):
    image = np.uint8(image * 255)

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels_im = cv2.connectedComponents(binary)

    bounding_boxes = []
    for label in range(1, num_labels):
        points = np.column_stack(np.where(labels_im == label))
        x, y, w, h = cv2.boundingRect(points)
        if w > 20 or h > 20:
            continue
        bounding_boxes.append((x, y, w, h))

    return len(bounding_boxes), np.array(bounding_boxes)


if __name__ == "__main__":
    vae = VaeAgent()
    vae.load_model()
    resnet = create_feature_extractor()

    x_test, y_test = data_loader.load_test_data()
    y_test_bin = [1 if x == 0 else 0 for x in y_test]

    f1scores, aurocs = [], []
    for i in range(10):
        start = time()
        gen_data = vae.predict(x_test)

        scores = []
        for x, x_hat in zip(x_test, gen_data):
            x_len, x_boxes = get_bounding_boxes(x)
            x_hat_len, x_hat_boxes = get_bounding_boxes(x_hat)

            x = apply_mask_for_each_box(x, x_boxes)
            x_hat = apply_mask_for_each_box(x_hat, x_hat_boxes)

            embed_x = extract_features(resnet, x)
            embed_x_hat = extract_features(resnet, x_hat)
            similarity = cosine_similarity(embed_x, embed_x_hat)[0][0]
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
                show_images_with_data(vae.class_names[i], class_images[i], class_scores[i], is_float=True)

        fpr, tpr, thresholds = roc_curve(y_test_bin, scores)
        show_roc_curve(fpr, tpr)
        auroc = roc_auc_score(y_test_bin, scores)

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
        print("AUROC      : %3f" % auroc)
        print()
        print(report)

        f1scores.append(f_score[0])
        aurocs.append(auroc)

    mean, variance = np.mean(aurocs), np.var(aurocs)
    print("AUROC :  ", aurocs)
    print(f"AUROC 평균: {mean}, 분산: {variance}")

    print("F1 scores :", f1scores)
    mean, variance = np.mean(f1scores), np.var(f1scores)
    print(f"F1 scores 평균: {mean}, 분산: {variance}")
