import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from data_prep.utils import TEST_IMAGES_DIR, TEST_LABELS_DIR
from train.utils import IMG_SIZE, BATCH_SIZE


# ===============================
# 이미지 전처리 (train과 동일)
# ===============================
def preprocess_image(path, color):
    channels = 1 if color == 'gray' else 3
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=channels, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    img = tf.cast(img, tf.float32) / 255.0
    if channels == 1:
        img = tf.expand_dims(img, axis=-1)
    return img


# ===============================
# Test Dataset 생성 (경로 기반)
# ===============================
def make_test_dataset(paths, color):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        lambda path: preprocess_image(path, color),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ===============================
# 테스트 함수
# ===============================
def test_model(selected_color, selected_strategy, model_path, epoch):
    """
    selected_color   : 'gray' or 'color'
    selected_strategy: 전략 번호 (로그용)
    model_path       : 학습된 모델(.h5) 경로
    epoch            : 테스트에 사용한 epoch
    """

    model_dir = os.path.dirname(model_path)
    results_txt_path = os.path.join(model_dir, f"results_epoch_{epoch:02d}.txt")
    confusion_matrix_path = os.path.join(model_dir, f"confusion_matrix_epoch_{epoch:02d}.png")
    test_files_txt_path = os.path.join(model_dir, f"test_files_epoch_{epoch:02d}.txt")


    # -------------------------------
    # 모델 로드
    # -------------------------------
    model = tf.keras.models.load_model(model_path)

    # -------------------------------
    # 테스트 이미지 경로 수집
    # -------------------------------
    img_ext = (".jpg", ".jpeg", ".png", ".bmp")
    img_paths = sorted([
        os.path.join(TEST_IMAGES_DIR, f)
        for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith(img_ext)
    ])

    if len(img_paths) == 0:
        raise ValueError("⚠ 테스트 이미지가 없습니다.")

    # -------------------------------
    # GT 라벨 로드
    # -------------------------------
    y_true = []
    used_files = []

    for img_path in img_paths:
        img_name = os.path.basename(img_path)

        label_path = os.path.join(
            TEST_LABELS_DIR,
            img_name.replace("image", "label").rsplit(".", 1)[0] + ".txt"
        )

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"라벨 파일 없음: {label_path}")

        with open(label_path, "r") as f:
            gt_label = int(f.readline().split()[0])

        y_true.append(gt_label)
        used_files.append((img_name, os.path.basename(label_path)))

    # -------------------------------
    # 예측
    # -------------------------------
    test_ds = make_test_dataset(img_paths, selected_color)
    y_prob = model.predict(test_ds, verbose=0).squeeze()
    y_pred = (y_prob > 0.5).astype(int)

    # -------------------------------
    # 성능 지표 계산
    # -------------------------------
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    
    # -------------------------------
    # 정답 / 오답 개수
    # -------------------------------
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    correct_count = tp + tn
    wrong_count   = fp + fn
    total_count   = correct_count + wrong_count

    # -------------------------------
    # 결과 저장 (TXT)
    # -------------------------------
    results_str = (
        f"=== Test Results (Epoch {epoch}) ===\n"
        f"Color      : {selected_color}\n"
        f"Strategy   : {selected_strategy}\n"
        f"Total      : {total_count}\n"
        f"Correct    : {correct_count}\n"
        f"Wrong      : {wrong_count}\n"
        f"\n"
        f"Accuracy   : {accuracy:.4f}\n"
        f"Precision  : {precision:.4f}\n"
        f"Recall     : {recall:.4f}\n"
        f"F1-score   : {f1:.4f}\n"
    )

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

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()

    print(results_str)
    print(f"✅ 결과 TXT 저장: {results_txt_path}")
    print(f"✅ Confusion Matrix 저장: {confusion_matrix_path}")
    
    with open(test_files_txt_path, "w", encoding="utf-8") as f:
        f.write("=== Test Files Used ===\n")
        f.write(f"파일 개수 : {len(used_files)}\n")
        f.write("-" * 40 + "\n")
        for img_name, label_name in used_files:
            f.write(f"{img_name}  |  {label_name}\n")

    print(f"✅ 테스트 파일 목록 저장: {test_files_txt_path}")

    # -------------------------------
    # 결과 반환
    # -------------------------------
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "total_count": total_count
    }
