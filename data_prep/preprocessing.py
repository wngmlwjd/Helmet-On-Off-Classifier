import os
import cv2
import numpy as np
import re

from data_prep.utils import (
    FILTERED_IMAGES_DIR, FILTERED_LABELS_DIR,
    OUTPUT_COLOR_DIRS,
    TARGET_SIZE,
    TARGET_ASPECT_RATIO,
    get_bbox_pixel_coords
)


def ensure_dirs(dir_dict):
    for key in dir_dict:
        os.makedirs(os.path.join(dir_dict[key], 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_dict[key], 'labels'), exist_ok=True)

def save_image_and_label(img_crop, cls, number, seq, out_img_dir, out_label_dir):
    """이미지 저장 + 클래스 값만 있는 라벨 저장"""
    img_name = f"image_{number}_{seq}.jpg"
    label_name = f"label_{number}_{seq}.txt"

    img_path = os.path.join(out_img_dir, img_name)
    cv2.imwrite(img_path, img_crop)

    label_path = os.path.join(out_label_dir, label_name)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(str(cls) + "\n")

def forced_scale(img, x_min, y_min, x_max, y_max):
    crop = img[y_min:y_max, x_min:x_max]
    return cv2.resize(crop, TARGET_SIZE)

def padded_scale(img, x_min, y_min, x_max, y_max):
    crop = img[y_min:y_max, x_min:x_max]
    h, w = crop.shape[:2]
    target_w, target_h = TARGET_SIZE
    target_aspect = target_w / target_h
    current_aspect = w / h
    
    if current_aspect > target_aspect:
        new_h = int(w / target_aspect)
        pad_vert = new_h - h
        pad_top = pad_vert // 2
        pad_bottom = pad_vert - pad_top
        padded = cv2.copyMakeBorder(crop, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        new_w = int(h * target_aspect)
        pad_horz = new_w - w
        pad_left = pad_horz // 2
        pad_right = pad_horz - pad_left
        padded = cv2.copyMakeBorder(crop, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    
    return cv2.resize(padded, TARGET_SIZE)

def aspect_aware_crop(img, x_min, y_min, x_max, y_max):
    h, w = img.shape[:2]
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    bbox_aspect = bbox_w / bbox_h

    if bbox_aspect > TARGET_ASPECT_RATIO:
        target_h = int(bbox_w / TARGET_ASPECT_RATIO)
        center_y = (y_min + y_max) // 2
        y_min = max(0, center_y - target_h // 2)
        y_max = min(h, center_y + target_h // 2)
    else:
        target_w = int(bbox_h * TARGET_ASPECT_RATIO)
        center_x = (x_min + x_max) // 2
        x_min = max(0, center_x - target_w // 2)
        x_max = min(w, center_x + target_w // 2)

    crop = img[y_min:y_max, x_min:x_max]
    return cv2.resize(crop, TARGET_SIZE)

def replicate_padded_scale(img, x_min, y_min, x_max, y_max):
    crop = img[y_min:y_max, x_min:x_max]
    h, w = crop.shape[:2]

    target_w, target_h = TARGET_SIZE
    target_aspect = target_w / target_h
    current_aspect = w / h

    if current_aspect > target_aspect:
        new_h = int(w / target_aspect)
        pad_vert = new_h - h
        pad_top = pad_vert // 2
        pad_bottom = pad_vert - pad_top
        padded = cv2.copyMakeBorder(
            crop,
            pad_top, pad_bottom, 0, 0,
            cv2.BORDER_REPLICATE
        )
    else:
        new_w = int(h * target_aspect)
        pad_horz = new_w - w
        pad_left = pad_horz // 2
        pad_right = pad_horz - pad_left
        padded = cv2.copyMakeBorder(
            crop,
            0, 0, pad_left, pad_right,
            cv2.BORDER_REPLICATE
        )

    return cv2.resize(padded, TARGET_SIZE)


def preprocess_dataset():
    ensure_dirs(OUTPUT_COLOR_DIRS)
    img_files = sorted(f for f in os.listdir(FILTERED_IMAGES_DIR) if f.lower().endswith(('.jpg','.png','.jpeg')))
    total_images = len(img_files)
    print(f"총 {total_images}장의 이미지 처리 시작...")

    for img_idx, img_name in enumerate(img_files, start=1):
        img_path = os.path.join(FILTERED_IMAGES_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 이미지 읽기 실패: {img_name}")
            continue

        # 이미지 번호 추출: image_123.jpg → 123
        number_match = re.search(r'_(\d+)\.', img_name)
        if not number_match:
            print(f"⚠️ 번호 추출 실패: {img_name}")
            continue
        number = number_match.group(1)
        label_name = f"label_{number}.txt"
        label_path = os.path.join(FILTERED_LABELS_DIR, label_name)
        
        if not os.path.isfile(label_path):
            print(f"⚠️ 라벨 파일 없음: {label_name}")
            continue

        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        
        print(f"[{img_idx}/{total_images}] 처리 중: {img_name} ({len(lines)}개의 라벨)")

        for seq, line in enumerate(lines, start=1):
            cls, x_min, y_min, x_max, y_max = get_bbox_pixel_coords(line, img.shape[1], img.shape[0])
            if cls is None:
                continue

            # 각 전처리 방식 별로 이미지와 라벨 저장
            save_image_and_label(forced_scale(img, x_min, y_min, x_max, y_max), cls, number, seq, os.path.join(OUTPUT_COLOR_DIRS['forced'], 'images'), os.path.join(OUTPUT_COLOR_DIRS['forced'], 'labels'))
            save_image_and_label(padded_scale(img, x_min, y_min, x_max, y_max), cls, number, seq, os.path.join(OUTPUT_COLOR_DIRS['padded'], 'images'), os.path.join(OUTPUT_COLOR_DIRS['padded'], 'labels'))
            save_image_and_label(aspect_aware_crop(img, x_min, y_min, x_max, y_max), cls, number, seq, os.path.join(OUTPUT_COLOR_DIRS['aware'], 'images'), os.path.join(OUTPUT_COLOR_DIRS['aware'], 'labels'))
            save_image_and_label(replicate_padded_scale(img, x_min, y_min, x_max, y_max), cls, number, seq, os.path.join(OUTPUT_COLOR_DIRS['replicate'], 'images'), os.path.join(OUTPUT_COLOR_DIRS['replicate'], 'labels'))

    print("모든 이미지 처리 완료!")


if __name__ == "__main__":
    preprocess_dataset()
