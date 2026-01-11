import os
from datetime import datetime

from data_prep.utils import TARGET_SIZE, SUB_DIRS

# ===============================
# 모델 저장 경로
# ===============================
BASE_MODEL_DIR = "./model"

def get_train_model_save_dir(color, strategy, today_str):
    base_dir = os.path.join(BASE_MODEL_DIR, color, SUB_DIRS[strategy])
    
    for i in range(1, 100):
        save_dir = os.path.join(base_dir, today_str + f"_{i:02d}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            return save_dir
    raise RuntimeError("모델 저장 경로 생성 실패")

# ===============================
# 학습 관련 상수
# ===============================
IMG_SIZE = TARGET_SIZE
BATCH_SIZE = 32
EPOCHS = 200
HELMET_CLASS_ID = 1