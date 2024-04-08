import cv2
import numpy as np
from time import time

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.data import data_loader
from src.model.vae_agent import VaeAgent
from src.visualization.image import show_images_with_data
from src.visualization.plot import show_score_density_plot, show_roc_curve
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

def create_feature_extractor():
    # ResNet50 모델을 이미지넷 가중치와 함께 불러옴, 최상위 분류 레이어는 포함하지 않음
    base_model = ResNet50(weights='imagenet', include_top=False, )

    # layer2[-1]과 layer3[-1]에 해당하는 레이어의 출력을 가져옴
    layer2_output = base_model.get_layer('conv3_block4_out').output  # ResNet50에서 layer2에 해당
    layer3_output = base_model.get_layer('conv4_block6_out').output  # ResNet50에서 layer3에 해당

    # 가져온 레이어의 출력을 플랫하게 만듦
    flattened_layer2_output = tf.keras.layers.Flatten()(layer2_output)
    flattened_layer3_output = tf.keras.layers.Flatten()(layer3_output)

    # 플랫한 출력을 결합
    concatenated_output = tf.keras.layers.Concatenate(axis=-1)([flattened_layer2_output, flattened_layer3_output])

    # 새로운 모델을 정의함. 이 모델은 ResNet50의 입력을 받아서, 수정된 출력을 내보냄
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


def get_bounding_boxes(image, size=64):
    image = np.uint8(image * 255)

    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append([x, y, x + w, y + h])

    length = len(bounding_boxes)
    if len(bounding_boxes) < size:
        bounding_boxes += [0, 0, 0, 0] * (size - len(bounding_boxes))
    bounding_boxes = bounding_boxes[:size]

    return length, np.array(bounding_boxes)

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

    print("F1 scores :", f1scores)
    print("AUROCs :  ", aurocs)