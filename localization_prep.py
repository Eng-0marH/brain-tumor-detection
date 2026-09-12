import os
import glob
import cv2
import yaml

from config import TRAIN_PATH, VAL_PATH, CLASS_NAMES, TUMOR_CLASS_IDS, LOCALIZATION_DATASET_PATH
from preprocessing import apply_clahe


def _get_images(folder):
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(images)


def _map_label_lines(label_path):
    
    #every tumor id mapped to a single class 0
    
    if not os.path.exists(label_path):
        return []

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    mapped_lines = []

    for line in lines:
        parts = line.split()
        class_id = int(float(parts[0]))

        if class_id in TUMOR_CLASS_IDS:
            mapped_lines.append("0 " + " ".join(parts[1:]))

    return mapped_lines


def _build_split(split_path, split_name, images_dir, labels_dir):
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    count = 0

    for class_id, class_name in CLASS_NAMES.items():
        source_images = os.path.join(split_path, class_name, "images")
        source_labels = os.path.join(split_path, class_name, "labels")

        for image_path in _get_images(source_images):
            image_name = os.path.basename(image_path)
            stem = os.path.splitext(image_name)[0]
            label_path = os.path.join(source_labels, stem + ".txt")

            mapped_lines = _map_label_lines(label_path)

            image = cv2.imread(image_path)
            if image is None:
                continue

            image = apply_clahe(image)
            cv2.imwrite(os.path.join(images_dir, image_name), image)

            with open(os.path.join(labels_dir, stem + ".txt"), "w") as f:
                for line in mapped_lines:
                    f.write(line + "\n")

            count += 1

    print(f"{split_name}: {count} images prepared for localization")
    return count


def build_localization_dataset():
    train_images_dir = os.path.join(LOCALIZATION_DATASET_PATH, "train", "images")
    train_labels_dir = os.path.join(LOCALIZATION_DATASET_PATH, "train", "labels")
    val_images_dir = os.path.join(LOCALIZATION_DATASET_PATH, "val", "images")
    val_labels_dir = os.path.join(LOCALIZATION_DATASET_PATH, "val", "labels")

    _build_split(TRAIN_PATH, "Train", train_images_dir, train_labels_dir)
    _build_split(VAL_PATH, "Val", val_images_dir, val_labels_dir)

    yaml_path = os.path.join(LOCALIZATION_DATASET_PATH, "data.yaml")

    yaml_content = {
        "train": train_images_dir,
        "val": val_images_dir,
        "names": {0: "Tumor"},
    }

    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"data.yaml written to: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    build_localization_dataset()
