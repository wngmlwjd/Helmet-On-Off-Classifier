import os
import shutil
import numpy as np
import cv2

FILTER_SIZE = (100, 100)  # 최소 bbox 크기 기준 (픽셀 기준)

# TARGET_SIZE = (107, 128)  # 목표 크기 (width, height)
TARGET_SIZE = (54, 64)  # 목표 크기 (width, height)
TARGET_ASPECT_RATIO = 0.8384  # 목표 종횡비 (너비/높이)

DATASET_DIR = "./dataset"

RAW_DATASET_DIR = DATASET_DIR + "/01_raw"
RAW_IMAGES_DIR = RAW_DATASET_DIR + "/images"
RAW_LABELS_DIR = RAW_DATASET_DIR + "/labels"

FILTERED_DATASET_DIR = DATASET_DIR + "/02_filtered"
FILTERED_IMAGES_DIR = FILTERED_DATASET_DIR + "/images"
FILTERED_LABELS_DIR = FILTERED_DATASET_DIR + "/labels"

PREPROCESSED_DIR = DATASET_DIR + "/03_preprocessed"
PREPROCESSED_COLOR_DIR = PREPROCESSED_DIR + "/color"
PREPROCESSED_GRAY_DIR = PREPROCESSED_DIR + "/gray"

SPLITED_DIR = DATASET_DIR + "/04_splited"

TRAIN_DATASET_DIR = DATASET_DIR + "/train"
TRAIN_IMAGES_DIR = TRAIN_DATASET_DIR + "/images"
TRAIN_LABELS_DIR = TRAIN_DATASET_DIR + "/labels"

TEST_DATASET_DIR = DATASET_DIR + "/test"
TEST_IMAGES_DIR = TEST_DATASET_DIR + "/images"
TEST_LABELS_DIR = TEST_DATASET_DIR + "/labels"

CUT_DIR = DATASET_DIR + "/cut"
CUT_IMAGES_DIR = CUT_DIR + "/images"
CUT_LABELS_DIR = CUT_DIR + "/labels"

SUB_DIRS = {
    1: '1. forced_scale',
    2: '2. padded_scale',
    3: '3. aspect_aware_crop',
    4: '4. replicate_padded_scale',
}

# DATASET_TYPES = {
#     'forced': "1. forced_scale",
#     'padded': "2. padded_scale",
#     'aware': "3. aspect_aware_crop",
#     'replicate': "4. replicate_padded_scale",
# }

# OUTPUT_COLOR_DIRS = {
#     'forced': PREPROCESSED_COLOR_DIR + '/1. forced_scale',
#     'padded': PREPROCESSED_COLOR_DIR + '/2. padded_scale',
#     'aware':  PREPROCESSED_COLOR_DIR + '/3. aspect_aware_crop',
#     'replicate': PREPROCESSED_COLOR_DIR + '/4. replicate_padded_scale',
# }
# OUTPUT_GRAY_DIRS = {
#     'forced': PREPROCESSED_GRAY_DIR + '/1. forced_scale',
#     'padded': PREPROCESSED_GRAY_DIR + '/2. padded_scale',
#     'aware':  PREPROCESSED_GRAY_DIR + '/3. aspect_aware_crop',
#     'replicate': PREPROCESSED_GRAY_DIR + '/4. replicate_padded_scale',
# }


def clamp_coordinates(x_min, y_min, x_max, y_max, w, h):
    """좌표를 이미지 경계 내로 클램핑합니다."""
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(w, int(x_max))
    y_max = min(h, int(y_max))
    return x_min, y_min, x_max, y_max


def get_bbox_pixel_coords(line, w, h, target_aspect=None):
    """
    YOLO 라벨 라인에서 픽셀 좌표를 계산합니다. 
    target_aspect가 주어지면, 해당 종횡비에 맞게 Bounding Box를 조정합니다 (Strategy 3).
    """
    nums = line.strip().split()
    if len(nums) < 5:
        return None, None, None, None, None

    cls = nums[0]
    x_center, y_center, width, height = map(float, nums[1:5])

    # Normalized BBox -> Pixel BBox
    x_min = (x_center - width / 2) * w
    y_min = (y_center - height / 2) * h
    x_max = (x_center + width / 2) * w
    y_max = (y_center + height / 2) * h

    # --- Aspect-Ratio Adjustment (Strategy 3 only) ---
    if target_aspect is not None and target_aspect > 0:
        current_w = x_max - x_min
        current_h = y_max - y_min
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # 현재 종횡비와 목표 종횡비를 비교하여 더 넓거나 더 긴 쪽을 기준으로 BBox 확장
        if current_w / current_h > target_aspect:
            # 현재 BBox가 목표보다 가로로 더 넓음 -> 세로를 늘려야 함
            target_h_pixels = current_w / target_aspect
            dy = (target_h_pixels - current_h) / 2
            y_min = center_y - target_h_pixels / 2
            y_max = center_y + target_h_pixels / 2
        elif current_w / current_h < target_aspect:
            # 현재 BBox가 목표보다 세로로 더 길음 -> 가로를 늘려야 함
            target_w_pixels = current_h * target_aspect
            dx = (target_w_pixels - current_w) / 2
            x_min = center_x - target_w_pixels / 2
            x_max = center_x + target_w_pixels / 2
            
    # 이미지 경계 내로 클램핑
    x_min, y_min, x_max, y_max = clamp_coordinates(x_min, y_min, x_max, y_max, w, h)
    
    return cls, x_min, y_min, x_max, y_max

