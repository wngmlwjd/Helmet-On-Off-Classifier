import os

from data_prep.utils import TARGET_SIZE

SELECTED_COLOR = 'gray'  # 'color' 또는 'gray' 선택
SELECTED_STRATEGY = 3   # 1, 2, 또는 3 선택
MODEL_NUM = 9
TEST_MODEL_DATE = '20251202_01'

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

TEST_IMAGE_DIR = os.path.join(
    PREPROCESSED_DIR,
    SELECTED_COLOR,
    "test",
    SUB_DIRS[SELECTED_STRATEGY],
    "images"
)
TEST_LABEL_DIR = os.path.join(
    PREPROCESSED_DIR,
    SELECTED_COLOR,
    "test",
    SUB_DIRS[SELECTED_STRATEGY],
    "labels"
)

# ===============================
# 모델 저장 경로
# ===============================
BASE_MODEL_DIR = "./model"

TEST_MODEL_SAVE_DIR = os.path.join(
    BASE_MODEL_DIR,
    SELECTED_COLOR,
    SUB_DIRS[SELECTED_STRATEGY],
    TEST_MODEL_DATE,
)

TEST_MODEL_PATH = os.path.join(TEST_MODEL_SAVE_DIR, f"epoch_{MODEL_NUM:02d}.h5")

# ===============================
# 성능 평가 관련 상수
# ===============================
IMG_SIZE = TARGET_SIZE

CONFUSION_MATRIX_SAVE_DIR = TEST_MODEL_SAVE_DIR
CONFUSION_MATRIX_SAVE_PATH = os.path.join(
    CONFUSION_MATRIX_SAVE_DIR,
    SELECTED_COLOR + "_" + str(SELECTED_STRATEGY) + f"_epoch{MODEL_NUM:02d}" + ".png",
)

RESULTS_TXT_SAVE_DIR = TEST_MODEL_SAVE_DIR
RESULTS_TXT_SAVE_PATH = os.path.join(
    TEST_MODEL_SAVE_DIR,
    SELECTED_COLOR + "_" + str(SELECTED_STRATEGY) + f"_epoch{MODEL_NUM:02d}" + ".txt",
)

# print(CONFUSION_MATRIX_SAVE_PATH)