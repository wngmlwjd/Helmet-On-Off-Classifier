import os
import tensorflow as tf
import csv
import numpy as np
from sklearn.model_selection import train_test_split

from data_prep.utils import TRAIN_DATASET_DIR, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, TEST_DATASET_DIR, TEST_IMAGES_DIR, TEST_LABELS_DIR
from train.utils import HELMET_CLASS_ID, IMG_SIZE, BATCH_SIZE, EPOCHS
from model.model_17 import main_cnn 

# ===============================
# 이미지 로드 함수
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
# Dataset 생성 함수
# ===============================
def make_train_val_dataset(paths, labels, color, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda path, label: (preprocess_image(path, color), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ===============================
# 학습 함수 (Single Train/Val)
# ===============================
def train_model(selected_color, selected_strategy, model_save_dir, val_split_ratio):
    os.makedirs(model_save_dir, exist_ok=True)

    # -------------------------------
    # YOLO 라벨 기반 이미지/라벨 리스트 생성
    # -------------------------------
    label_files = sorted([
        f for f in os.listdir(TRAIN_LABELS_DIR)
        if f.startswith("label_") and f.endswith(".txt")
    ])

    img_paths, labels = [], []

    for label_file in label_files:
        base = label_file.replace("label_", "").replace(".txt", "")
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            fname = os.path.join(TRAIN_IMAGES_DIR, f"image_{base}{ext}")
            if os.path.exists(fname):
                img_path = fname
                break
        if img_path is None:
            continue

        has_helmet = 0
        with open(os.path.join(TRAIN_LABELS_DIR, label_file), "r") as f:
            for line in f:
                if not line.strip():
                    continue
                class_id = int(line.split()[0])
                if class_id == HELMET_CLASS_ID:
                    has_helmet = 1
                    break

        img_paths.append(img_path)
        labels.append(has_helmet)

    if len(labels) == 0:
        raise ValueError("⚠ 데이터셋이 비어 있습니다.")

    img_paths = np.array(img_paths)
    labels = np.array(labels)

    # -------------------------------
    # Train / Validation 분할
    # -------------------------------
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        img_paths,
        labels,
        test_size=val_split_ratio,
        stratify=labels,
    )

    train_ds = make_train_val_dataset(train_paths.tolist(), train_labels.tolist(), selected_color, shuffle=True)
    val_ds   = make_train_val_dataset(val_paths.tolist(), val_labels.tolist(), selected_color, shuffle=False)

    # -------------------------------
    # 모델 생성
    # -------------------------------
    tf.keras.backend.clear_session()
    model = main_cnn(selected_color)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # -------------------------------
    # Callbacks
    # -------------------------------
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_save_dir, "best_model.h5"),
        monitor="val_accuracy",      # 또는 "val_loss"
        mode="max",                  # val_loss 쓰면 "min"
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    best_result = {}
    
    class HistoryCSVCallback(tf.keras.callbacks.Callback):
        def on_train_end(self, logs=None):
            hist = self.model.history.history
            best_val_acc = max(hist['val_accuracy'])
            best_epoch = hist['val_accuracy'].index(best_val_acc) + 1

            csv_path = os.path.join(model_save_dir, "history.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([f"Best Val Accuracy: {best_val_acc:.4f}", f"Epoch: {best_epoch}"])
                writer.writerow([])
                keys = list(hist.keys())
                writer.writerow(["epoch"] + keys)
                for i in range(len(hist[keys[0]])):
                    writer.writerow([i+1] + [hist[k][i] for k in keys])

            print("✅ history.csv 저장 완료")
            
            best_result["best_epoch"] = best_epoch
            best_result["best_val_acc"] = best_val_acc
    
    # -------------------------------
    # 상수 기록
    # -------------------------------
    with open(os.path.join(model_save_dir, "training_constants.txt"), "w", encoding="utf-8") as f:
        f.write("학습 관련 상수 및 하이퍼파라미터\n")
        f.write("==============================\n")
        f.write(f"IMG_SIZE: {IMG_SIZE}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"VAL_RATIO: {val_split_ratio}\n")
        f.write(f"SELECTED_COLOR: {selected_color}\n")
        f.write(f"SELECTED_STRATEGY: {selected_strategy}\n")

    # -------------------------------
    # 학습
    # -------------------------------
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            checkpoint_cb,
            earlystop_cb,
            reduce_lr_cb,
            HistoryCSVCallback()
        ]
    )

    return best_result