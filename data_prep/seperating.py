import os
import shutil
import random

from data_prep.utils import PREPROCESSED_DIR, SUB_DIRS, SPLITED_DIR, TRAIN_DATASET_DIR, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, TEST_DATASET_DIR, TEST_IMAGES_DIR, TEST_LABELS_DIR


def split_fold(color, strategy, n_folds):
    # 1) 기준이 될 이미지 목록 생성
    img_dir = os.path.join(PREPROCESSED_DIR, color, SUB_DIRS[strategy], "images")
    img_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg')))
    
    random.shuffle(img_files)
    n = int(len(img_files) / n_folds)
    
    if os.path.exists(SPLITED_DIR):
        shutil.rmtree(SPLITED_DIR)
    os.makedirs(SPLITED_DIR, exist_ok=True)
    
    fold_num = 1
    for idx, img_file in enumerate(img_files):
        os.makedirs(os.path.join(PREPROCESSED_DIR, color, SUB_DIRS[strategy], "images"), exist_ok=True)
        os.makedirs(os.path.join(SPLITED_DIR, f"fold_{fold_num:02d}", "images"), exist_ok=True)
        os.makedirs(os.path.join(PREPROCESSED_DIR, color, SUB_DIRS[strategy], "labels"), exist_ok=True)
        os.makedirs(os.path.join(SPLITED_DIR, f"fold_{fold_num:02d}", "labels"), exist_ok=True)
        
        src_img_path = os.path.join(PREPROCESSED_DIR, color, SUB_DIRS[strategy], "images", img_file)
        dst_img_path = os.path.join(SPLITED_DIR, f"fold_{fold_num:02d}", "images", img_file)
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)

        label_file = img_file.replace("image", "label").replace(".jpg", ".txt")
        src_label_path = os.path.join(PREPROCESSED_DIR, color, SUB_DIRS[strategy], "labels", label_file)
        dst_label_path = os.path.join(SPLITED_DIR, f"fold_{fold_num:02d}", "labels", label_file)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
            
        if (idx + 1) % n == 0 and fold_num < n_folds:
            fold_num += 1
            
def split_train_test(test_fold_num):
    if os.path.exists(TRAIN_DATASET_DIR):
        shutil.rmtree(TRAIN_DATASET_DIR)
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
    if os.path.exists(TEST_DATASET_DIR):
        shutil.rmtree(TEST_DATASET_DIR)
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(TEST_LABELS_DIR, exist_ok=True)
    
    for fold_num in range (len(os.listdir(SPLITED_DIR))):
        fold_num += 1
        src_dir = os.path.join(SPLITED_DIR, f"fold_{fold_num:02d}")
        src_img_dir = os.path.join(src_dir, "images")
        src_label_dir = os.path.join(src_dir, "labels")
        
        if fold_num == test_fold_num:
            for img_file in os.listdir(src_img_dir):
                shutil.copy2(os.path.join(src_img_dir, img_file), os.path.join(TEST_IMAGES_DIR, img_file))
                label_file = img_file.replace("image", "label").replace(".jpg", ".txt")
                shutil.copy2(os.path.join(src_label_dir, label_file), os.path.join(TEST_LABELS_DIR, label_file))
        else:
            for img_file in os.listdir(src_img_dir):
                shutil.copy2(os.path.join(src_img_dir, img_file), os.path.join(TRAIN_IMAGES_DIR, img_file))
                label_file = img_file.replace("image", "label").replace(".jpg", ".txt")
                shutil.copy2(os.path.join(src_label_dir, label_file), os.path.join(TRAIN_LABELS_DIR, label_file))