# 바운딩 박스 그대로 자르기

import os
import cv2
from data_prep.utils import CUT_IMAGES_DIR, CUT_LABELS_DIR, RAW_IMAGES_DIR, RAW_LABELS_DIR

os.makedirs(CUT_IMAGES_DIR, exist_ok=True)
os.makedirs(CUT_LABELS_DIR, exist_ok=True)

files = sorted(os.listdir(RAW_LABELS_DIR))

for idx, file in enumerate(files, start=1):
    label_file = f"label_{idx}.txt"
    
    print(label_file)
    
    image_name = label_file.replace("label_", "image_").replace(".txt", ".jpg")
    image_path = os.path.join(RAW_IMAGES_DIR, image_name)
    label_path = os.path.join(RAW_LABELS_DIR, label_file)

    if not os.path.exists(image_path):
        print(f"{image_path} 파일이 없습니다. 스킵")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"{image_name} 이미지 읽기 실패. 스킵")
        continue

    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for crop_idx, line in enumerate(lines):
        nums = line.strip().split()
        if len(nums) != 5:
            continue

        cls = nums[0]
        x_center, y_center, width, height = map(float, nums[1:5])

        x_min = int((x_center - width / 2) * w)
        y_min = int((y_center - height / 2) * h)
        x_max = int((x_center + width / 2) * w)
        y_max = int((y_center + height / 2) * h)

        # 좌표 클램핑
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w - 1, x_max)
        y_max = min(h - 1, y_max)

        crop = img[y_min:y_max, x_min:x_max]
        
        # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) # 흑백 변환

        crop_img_name = f"{os.path.splitext(image_name)[0]}_crop{crop_idx+1}.jpg"
        crop_img_path = os.path.join(CUT_IMAGES_DIR, crop_img_name)
        cv2.imwrite(crop_img_path, crop)

        crop_h, crop_w = crop.shape[:2]
        aspect_ratio = crop_w / crop_h if crop_h != 0 else 0

        crop_label_name = f"{os.path.splitext(label_file)[0]}_crop{crop_idx+1}.txt"
        crop_label_path = os.path.join(CUT_LABELS_DIR, crop_label_name)

        with open(crop_label_path, "w") as lf:
            lf.write(f"{cls} {crop_h} {crop_w} {aspect_ratio:.6f}\n")

print("크롭 이미지와 라벨 저장 완료!")
