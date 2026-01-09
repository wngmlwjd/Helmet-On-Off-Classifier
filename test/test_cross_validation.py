import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from test.utils import IMG_SIZE, TEST_IMAGE_DIR, TEST_LABEL_DIR


# ===============================
# 전처리 함수 (학습과 동일)
# ===============================
def preprocess(img, selected_color):
    """
    img: BGR 이미지 (cv2.imread)
    IMG_SIZE: (W, H)
    """
    if selected_color == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]))
        img = img[..., np.newaxis]  # (H, W, 1)
    else:
        img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]))

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W, C)
    return img


# ===============================
# 테스트 함수
# ===============================
def test_model(
    selected_color,
    selected_strategy,   # 기록용 (로직에는 영향 없음)
    model_path,
    epoch,
    set_idx=None,
    fold_idx=None,
    results_txt_dir="./results",
    confusion_matrix_dir="./confusion_matrix"
):
    """
    selected_color   : 'gray' or 'color'
    selected_strategy: 전처리 전략 번호 (기록용)
    model_path       : 테스트할 모델(.h5) 경로
    epoch            : 모델 epoch 번호 (파일명/로그용)
    """

    os.makedirs(results_txt_dir, exist_ok=True)
    os.makedirs(confusion_matrix_dir, exist_ok=True)

    results_txt_path = os.path.join(
        results_txt_dir,
        f"results_color_{selected_color}_strategy_{selected_strategy}_epoch_{epoch}.txt"
    )
    confusion_matrix_path = os.path.join(
        confusion_matrix_dir,
        f"cm_color_{selected_color}_strategy_{selected_strategy}_epoch_{epoch}.png"
    )

    # -------------------------------
    # 모델 로드
    # -------------------------------
    model = tf.keras.models.load_model(model_path)

    # -------------------------------
    # 테스트 이미지 수집
    # -------------------------------
    img_ext = (".jpg", ".jpeg", ".png", ".bmp")
    file_list = sorted([
        f for f in os.listdir(TEST_IMAGE_DIR)
        if f.lower().endswith(img_ext)
    ])

    if len(file_list) == 0:
        raise RuntimeError("⚠ 테스트 이미지가 없습니다.")

    y_true, y_pred = [], []

    # -------------------------------
    # 추론
    # -------------------------------
    for file_name in file_list:
        img_path = os.path.join(TEST_IMAGE_DIR, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        label_path = os.path.join(
            TEST_LABEL_DIR,
            file_name.replace("image", "label").rsplit(".", 1)[0] + ".txt"
        )
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            gt_label = int(f.readline().strip().split()[0])

        pred = model.predict(
            preprocess(img, selected_color),
            verbose=0
        )

        pred_prob = float(pred.squeeze())
        pred_label = int(pred_prob > 0.5)

        y_true.append(gt_label)
        y_pred.append(pred_label)

    # -------------------------------
    # 성능 지표 계산
    # -------------------------------
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    results_str = (
        f"=== Test Results ===\n"
        f"Color      : {selected_color}\n"
        f"Strategy   : {selected_strategy}\n"
        f"Set        : {set_idx}\n"
        f"Fold       : {fold_idx}\n"
        f"Epoch      : {epoch}\n"
        f"{'-'*40}\n"
        f"Accuracy   : {accuracy:.4f}\n"
        f"Precision  : {precision:.4f}\n"
        f"Recall     : {recall:.4f}\n"
        f"F1-score   : {f1:.4f}\n"
    )

    # -------------------------------
    # TXT 저장
    # -------------------------------
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(results_str)

    # -------------------------------
    # Confusion Matrix 저장
    # -------------------------------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks([0, 1], ["No Helmet", "Helmet"])
    plt.yticks([0, 1], ["No Helmet", "Helmet"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"{selected_color} | strategy {selected_strategy} | epoch {epoch}")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()

    # -------------------------------
    # 출력
    # -------------------------------
    print(results_str)
    print(f"✅ 결과 TXT 저장: {results_txt_path}")
    print(f"✅ Confusion Matrix 저장: {confusion_matrix_path}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

