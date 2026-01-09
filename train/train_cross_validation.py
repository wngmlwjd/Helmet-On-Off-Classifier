import os
import tensorflow as tf
import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import train.utils as U
from model.model_4 import main_cnn  # CNN 모델 불러오기

# ===============================
# 이미지 로드 함수
# ===============================
def load_image(path, label):
    channels = 1 if U.SELECTED_COLOR == 'gray' else 3
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=channels, expand_animations=False)
    img = tf.image.resize(img, (U.IMG_SIZE[1], U.IMG_SIZE[0]))
    img = tf.cast(img, tf.float32) / 255.0
    if channels == 1:
        img = tf.expand_dims(img, axis=-1)
    return img, label

# ===============================
# Dataset 생성 함수
# ===============================
def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(U.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ===============================
# 학습 함수
# ===============================
def train_model(selected_color, selected_strategy, model_save_dir, n_folds=U.N_FOLDS):
    # utils 기준으로만 사용
    U.SELECTED_COLOR = selected_color
    U.SELECTED_STRATEGY = selected_strategy

    os.makedirs(model_save_dir, exist_ok=True)

    # -------------------------------
    # YOLO 라벨 기반 이미지/라벨 리스트 생성
    # -------------------------------
    label_files = sorted([
        f for f in os.listdir(U.TRAIN_LABEL_DIR)
        if f.startswith("label_") and f.endswith(".txt")
    ])

    img_paths, labels = [], []

    for label_file in label_files:
        base = label_file.replace("label_", "").replace(".txt", "")
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            fname = os.path.join(U.TRAIN_IMAGE_DIR, f"image_{base}{ext}")
            if os.path.exists(fname):
                img_path = fname
                break
        if img_path is None:
            continue

        has_helmet = 0
        with open(os.path.join(U.TRAIN_LABEL_DIR, label_file), "r") as f:
            for line in f:
                if not line.strip():
                    continue
                class_id = int(line.split()[0])
                if class_id == U.HELMET_CLASS_ID:
                    has_helmet = 1
                    break

        img_paths.append(img_path)
        labels.append(has_helmet)

    if len(labels) == 0:
        raise ValueError("⚠ 데이터셋이 비어 있습니다.")

    img_paths = np.array(img_paths)
    labels = np.array(labels)

    # -------------------------------
    # Stratified K-Fold
    # -------------------------------
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=U.SEED
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(img_paths, labels), start=1):
        print(f"\n==============================")
        print(f"🚀 Fold {fold}/{n_folds} 학습 시작")
        print(f"==============================")

        fold_dir = os.path.join(model_save_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_paths = img_paths[train_idx].tolist()
        val_paths   = img_paths[val_idx].tolist()
        train_labels = labels[train_idx].tolist()
        val_labels   = labels[val_idx].tolist()

        train_ds = make_dataset(train_paths, train_labels, shuffle=True)
        val_ds   = make_dataset(val_paths, val_labels, shuffle=False)

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
            filepath=os.path.join(fold_dir, "epoch_{epoch:02d}.h5"),
            save_weights_only=False,
            save_freq="epoch"
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

        class HistoryCSVCallback(tf.keras.callbacks.Callback):
            def on_train_end(self, logs=None):
                hist = self.model.history.history
                best_val_acc = max(hist['val_accuracy'])
                best_epoch = hist['val_accuracy'].index(best_val_acc) + 1

                csv_path = os.path.join(fold_dir, "history.csv")
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([f"Best Val Accuracy: {best_val_acc:.4f}", f"Epoch: {best_epoch}"])
                    writer.writerow([])
                    keys = list(hist.keys())
                    writer.writerow(["epoch"] + keys)
                    for i in range(len(hist[keys[0]])):
                        writer.writerow([i+1] + [hist[k][i] for k in keys])

                print(f"✅ Fold {fold} history 저장 완료")

                fold_results.append({
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc
                })

        # -------------------------------
        # 상수 기록
        # -------------------------------
        with open(os.path.join(fold_dir, "training_constants.txt"), "w", encoding="utf-8") as f:
            f.write("학습 관련 상수 및 하이퍼파라미터\n")
            f.write("==============================\n")
            f.write(f"FOLD: {fold}/{n_folds}\n")
            f.write(f"IMG_SIZE: {U.IMG_SIZE}\n")
            f.write(f"BATCH_SIZE: {U.BATCH_SIZE}\n")
            f.write(f"EPOCHS: {U.EPOCHS}\n")
            f.write(f"SEED: {U.SEED}\n")
            f.write(f"SELECTED_COLOR: {U.SELECTED_COLOR}\n")
            f.write(f"SELECTED_STRATEGY: {U.SELECTED_STRATEGY}\n")

        # -------------------------------
        # 학습
        # -------------------------------
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=U.EPOCHS,
            callbacks=[
                checkpoint_cb,
                earlystop_cb,
                reduce_lr_cb,
                HistoryCSVCallback()
            ]
        )

    # -------------------------------
    # 전체 Fold 요약 출력
    # -------------------------------
    print("\n📊 Cross Validation 결과 요약")
    for r in fold_results:
        print(f"Fold {r['fold']} → Best Val Acc: {r['best_val_acc']:.4f} (Epoch {r['best_epoch']})")

    return fold_results

def run_all_trainings():
    colors = ["gray", "color"]
    strategies = [1, 2, 3, 4]

    for color in colors:
        for strategy in strategies:
            print("\n======================================")
            print(f"🎯 학습 시작: COLOR={color}, STRATEGY={strategy}")
            print("======================================")

            # utils 전역 값 갱신
            U.SELECTED_COLOR = color
            U.SELECTED_STRATEGY = strategy

            U.TRAIN_IMAGE_DIR = os.path.join(
                U.PREPROCESSED_DIR,
                color,
                "train",
                U.SUB_DIRS[strategy],
                "images"
            )
            U.TRAIN_LABEL_DIR = os.path.join(
                U.PREPROCESSED_DIR,
                color,
                "train",
                U.SUB_DIRS[strategy],
                "labels"
            )

            model_save_dir = U.get_train_model_save_dir(color, strategy)

            print(f"📂 IMAGE DIR: {U.TRAIN_IMAGE_DIR}")
            print(f"📂 LABEL DIR: {U.TRAIN_LABEL_DIR}")
            print(f"💾 SAVE DIR : {model_save_dir}")

            train_model(
                selected_color=color,
                selected_strategy=strategy,
                model_save_dir=model_save_dir,
                n_folds=U.N_FOLDS
            )
            
# if __name__ == "__main__":
#     run_all_trainings()