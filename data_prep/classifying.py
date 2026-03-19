'''
클래스 별 분류
dataset/03_preprocessed 데이터 사용
'''

import os
import shutil
# from data_prep.utils import PREPROCESSED_IMAGES_DIR, PREPROCESSED_LABELS_DIR

PREPROCESSED_IMAGES_DIR = '/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/dataset/03_preprocessed/gray/2. padded_scale/images'
PREPROCESSED_LABELS_DIR = '/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/dataset/03_preprocessed/gray/2. padded_scale/labels'

# 분류 결과를 저장할 루트 디렉토리
CLASSIFIED_IMAGES_DIR = "dataset/classified/images"
CLASSIFIED_LABELS_DIR = "dataset/classified/labels"

files = sorted(os.listdir(PREPROCESSED_LABELS_DIR))

for label_file in files:
    label_path = os.path.join(PREPROCESSED_LABELS_DIR, label_file)

    with open(label_path, "r") as f:
        line = f.readline().strip()

    if not line:
        print(f"{label_file} 내용 없음. 스킵")
        continue

    cls = line.split()[0]

    # 클래스별 폴더 생성
    cls_img_dir = os.path.join(CLASSIFIED_IMAGES_DIR, cls)
    cls_lbl_dir = os.path.join(CLASSIFIED_LABELS_DIR, cls)
    os.makedirs(cls_img_dir, exist_ok=True)
    os.makedirs(cls_lbl_dir, exist_ok=True)

    # 대응하는 이미지 파일 찾기 (확장자 .jpg 가정)
    image_file = label_file.replace("label_", "image_").replace(".txt", ".jpg")
    image_path = os.path.join(PREPROCESSED_IMAGES_DIR, image_file)

    if not os.path.exists(image_path):
        print(f"{image_path} 이미지 없음. 라벨만 복사")
    else:
        shutil.copy2(image_path, os.path.join(cls_img_dir, image_file))

    shutil.copy2(label_path, os.path.join(cls_lbl_dir, label_file))

    print(f"[class {cls}] {image_file} / {label_file} 저장 완료")

print("클래스별 분류 저장 완료!")