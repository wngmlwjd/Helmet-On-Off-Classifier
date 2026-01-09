import os
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split

from train.utils import SELECTED_COLOR, SELECTED_STRATEGY, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, HELMET_CLASS_ID, SEED
from model.model_10 import main_cnn  # CNN 모델 불러오기

# ===============================
# 이미지 로드 함수
# ===============================
def load_image(path, label):
    channels = 1 if SELECTED_COLOR == 'gray' else 3
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=channels, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    img = tf.cast(img, tf.float32) / 255.0
    if channels == 1:
        img = tf.expand_dims(img, axis=-1)  # (H, W, 1)
    return img, label

# ===============================
# Dataset 생성 함수
# ===============================
def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ===============================
# 학습 함수
# ===============================
def train_model(selected_color, selected_strategy, model_save_dir):
    global SELECTED_COLOR, SELECTED_STRATEGY
    SELECTED_COLOR = selected_color
    SELECTED_STRATEGY = selected_strategy
    TRAIN_MODEL_SAVE_DIR = model_save_dir
    os.makedirs(TRAIN_MODEL_SAVE_DIR, exist_ok=True)

    # -------------------------------
    # YOLO 라벨 기반 이미지/라벨 리스트 생성
    # -------------------------------
    label_files = sorted([f for f in os.listdir(TRAIN_LABEL_DIR) if f.startswith("label_") and f.endswith(".txt")])
    img_paths, labels = [], []

    for label_file in label_files:
        base = label_file.replace("label_", "").replace(".txt", "")
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            fname = os.path.join(TRAIN_IMAGE_DIR, f"image_{base}{ext}")
            if os.path.exists(fname):
                img_path = fname
                break
        if img_path is None:
            continue

        has_helmet = 0
        with open(os.path.join(TRAIN_LABEL_DIR, label_file), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                if class_id == HELMET_CLASS_ID:
                    has_helmet = 1
                    break

        img_paths.append(img_path)
        labels.append(has_helmet)

    if len(labels) == 0:
        raise ValueError("⚠ 데이터셋이 비어 있습니다. 경로와 파일 확인 필요")

    # -------------------------------
    # train / val split (9:1)
    # -------------------------------
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=0.1, random_state=SEED, stratify=labels
    )

    train_ds = make_dataset(train_paths, train_labels, shuffle=True)
    val_ds   = make_dataset(val_paths, val_labels, shuffle=False)

    # -------------------------------
    # 모델 생성 & 학습
    # -------------------------------
    model = main_cnn(selected_color)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 체크포인트 콜백
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(TRAIN_MODEL_SAVE_DIR, "epoch_{epoch:02d}.h5"),
        save_weights_only=False,
        save_freq="epoch"
    )

    # CSV 기록용 콜백
    class HistoryCSVCallback(tf.keras.callbacks.Callback):
        def on_train_end(self, logs=None):
            hist = self.model.history.history
            best_val_acc = max(hist['val_accuracy'])
            best_epoch = hist['val_accuracy'].index(best_val_acc) + 1
            csv_path = os.path.join(TRAIN_MODEL_SAVE_DIR, "history.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 최고 검증 정확도 맨 앞에 기록
                writer.writerow([f"Best Val Accuracy: {best_val_acc:.4f}", f"Epoch: {best_epoch}"])
                writer.writerow([])  # 빈 줄
                # 헤더 작성
                keys = list(hist.keys())
                writer.writerow(["epoch"] + keys)
                for i in range(len(hist[keys[0]])):
                    writer.writerow([i+1] + [hist[k][i] for k in keys])
            print(f"✅ History CSV 저장 완료: {csv_path}")

    # 학습 상수 TXT 저장
    constants_txt_path = os.path.join(TRAIN_MODEL_SAVE_DIR, "training_constants.txt")
    with open(constants_txt_path, "w", encoding="utf-8") as f:
        f.write("학습 관련 상수 및 하이퍼파라미터\n")
        f.write("==============================\n")
        f.write(f"TRAIN_IMAGE_DIR: {TRAIN_IMAGE_DIR}\n")
        f.write(f"TRAIN_LABEL_DIR: {TRAIN_LABEL_DIR}\n")
        f.write(f"TRAIN_MODEL_SAVE_DIR: {TRAIN_MODEL_SAVE_DIR}\n")
        f.write(f"IMG_SIZE: {IMG_SIZE}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"HELMET_CLASS_ID: {HELMET_CLASS_ID}\n")
        f.write(f"SEED: {SEED}\n")
        f.write(f"SELECTED_COLOR: {SELECTED_COLOR}\n")
        f.write(f"SELECTED_STRATEGY: {SELECTED_STRATEGY}\n")
        f.write(f"Callbacks: ModelCheckpoint\n")
    print(f"✅ 학습 상수 TXT 저장 완료: {constants_txt_path}")

    # 학습 실행
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, HistoryCSVCallback()]
    )

    best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1
    print(f"▶ 학습 완료 (최고 검증 정확도 epoch: {best_epoch})")
    return best_epoch
