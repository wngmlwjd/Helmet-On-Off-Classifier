import os
import shutil
import random

from data_prep.utils import OUTPUT_COLOR_DIRS, OUTPUT_GRAY_DIRS, DATASET_TYPES

# train/test 비율
TRAIN_RATIO = 0.9

def split_train_test_uniform(color_dirs, gray_dirs, output_root_color, output_root_gray):
    """
    모든 데이터셋(color/gray 각 3종류)에 동일한 train/test 분할 적용
    """
    # 1) 기준이 될 이미지 목록 생성 (color forced_scale 기준)
    base_dir = list(color_dirs.values())[0]  # 첫 번째 컬러 폴더 사용
    img_dir = os.path.join(base_dir, "images")
    img_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg')))
    
    random.shuffle(img_files)
    n_train = int(len(img_files) * TRAIN_RATIO)
    train_files = img_files[:n_train]
    test_files = img_files[n_train:]
    
    print(f"총 {len(img_files)}개 이미지 → train: {len(train_files)}, test: {len(test_files)}")

    # 2) 모든 컬러/그레이 데이터셋에 동일하게 적용
    for dataset_type, dirs_dict in [("color", color_dirs), ("gray", gray_dirs)]:
        output_root = output_root_color if dataset_type=="color" else output_root_gray
        for subset_type, dir_path in dirs_dict.items():
            img_dir = os.path.join(dir_path, "images")
            label_dir = os.path.join(dir_path, "labels")

            for split_name, files in zip(["train", "test"], [train_files, test_files]):
                out_img_dir = os.path.join(output_root, split_name, DATASET_TYPES[subset_type], "images")
                out_label_dir = os.path.join(output_root, split_name, DATASET_TYPES[subset_type], "labels")
                os.makedirs(out_img_dir, exist_ok=True)
                os.makedirs(out_label_dir, exist_ok=True)

                for img_file in files:
                    src_img_path = os.path.join(img_dir, img_file)
                    dst_img_path = os.path.join(out_img_dir, img_file)
                    if os.path.exists(src_img_path):
                        shutil.copy2(src_img_path, dst_img_path)

                    label_file = img_file.replace("image", "label").replace(".jpg", ".txt")
                    src_label_path = os.path.join(label_dir, label_file)
                    dst_label_path = os.path.join(out_label_dir, label_file)
                    if os.path.exists(src_label_path):
                        shutil.copy2(src_label_path, dst_label_path)

                print(f"✅ {dataset_type} {subset_type} → {split_name}: {len(files)}개 파일 복사 완료")

if __name__ == "__main__":
    random.seed(42)  # 재현성

    print("=== Color/Gray 데이터셋 동일 train/test 분리 ===")
    split_train_test_uniform(
        OUTPUT_COLOR_DIRS,
        OUTPUT_GRAY_DIRS,
        output_root_color="dataset/preprocessed/color",
        output_root_gray="dataset/preprocessed/gray"
    )

    print("\n모든 데이터셋 train/test 분리 완료!")
