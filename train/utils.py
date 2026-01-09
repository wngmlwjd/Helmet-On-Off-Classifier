import os
from datetime import datetime

from data_prep.utils import TARGET_SIZE

SELECTED_COLOR = 'gray'  # 'color' 또는 'gray' 선택
SELECTED_STRATEGY = 3  # 1 ~ 4 선택

# ===============================
# 데이터셋 경로
# ===============================
DATASET_DIR = "./dataset"

PREPROCESSED_DIR = DATASET_DIR + "/preprocessed"

SUB_DIRS = {
    1: '1. forced_scale',
    2: '2. padded_scale',
    3: '3. aspect_aware_crop',
    4: '4. replicate_padded_scale',
}

TRAIN_IMAGE_DIR = os.path.join(
    PREPROCESSED_DIR,
    SELECTED_COLOR,
    "train",
    SUB_DIRS[SELECTED_STRATEGY],
    "images"
)
TRAIN_LABEL_DIR = os.path.join(
    PREPROCESSED_DIR,
    SELECTED_COLOR,
    "train",
    SUB_DIRS[SELECTED_STRATEGY],
    "labels"
)

# ===============================
# 모델 저장 경로
# ===============================
BASE_MODEL_DIR = "./model"

def get_train_model_save_dir(color, strategy):
    today_str = datetime.today().strftime("%Y%m%d")
    base_dir = os.path.join(BASE_MODEL_DIR, color, SUB_DIRS[strategy])
    
    for i in range(1, 100):
        save_dir = os.path.join(base_dir, today_str + f"_{i:02d}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            return save_dir
    raise RuntimeError("모델 저장 경로 생성 실패")


# print(f"모델 저장 경로: {MODEL_SAVE_DIR}")

# ===============================
# 학습 관련 상수
# ===============================
IMG_SIZE = TARGET_SIZE
BATCH_SIZE = 32
EPOCHS = 200
HELMET_CLASS_ID = 1
SEED = 42

# ===============================
# 학습(Cross Validation) 관련 상수
# ===============================
N_FOLDS = 9
RUN_TIMES = 10 # 전체 Cross Validation 반복 횟수

# # ===============================
# # 특정 epoch 모델 별도 저장
# # ===============================
# TARGET_EPOCH = 16
# EPOCH_FILE = os.path.join(MODEL_SAVE_DIR, f"epoch_{TARGET_EPOCH:02d}.h5")
# TARGET_MODEL_FILE = os.path.join(MODEL_SAVE_DIR, f"helmet_main_epoch{TARGET_EPOCH}.h5")
