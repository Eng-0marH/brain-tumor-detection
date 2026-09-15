import os
import glob
import cv2

from config import (
    TRAIN_PATH, VAL_PATH, CLASS_NAMES, TUMOR_CLASS_IDS,
    CLASSIFIER_DATASET_PATH, CROP_PADDING_RATIO,
)
from preprocessing import apply_clahe, pad_and_clip
from reporting import print_header


def _get_images(folder):
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(images)


def _crop_split(split_path, split_name, output_root):
    for class_id in TUMOR_CLASS_IDS:
        os.makedirs(os.path.join(output_root, CLASS_NAMES[class_id]), exist_ok=True)

    total = 0

    print(f"\n{split_name} -> {output_root}")

    for class_id in sorted(TUMOR_CLASS_IDS):
        class_name = CLASS_NAMES[class_id]

        images_path = os.path.join(split_path, class_name, "images")
        labels_path = os.path.join(split_path, class_name, "labels")
        output_path = os.path.join(output_root, class_name)

        crop_count = 0

        for image_path in _get_images(images_path):
            stem = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(labels_path, stem + ".txt")

            if not os.path.exists(label_path):
                continue

            with open(label_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                continue

            image = cv2.imread(image_path)
            if image is None:
                continue

            image = apply_clahe(image)

            height, width = image.shape[:2]

            for box_index, line in enumerate(lines):
                parts = line.split()

                if len(parts) != 5:
                    continue

                x_center, y_center, box_width, box_height = map(float, parts[1:])

                x_center *= width
                y_center *= height
                box_width *= width
                box_height *= height

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                x1, y1, x2, y2 = pad_and_clip(
                    x1, y1, x2, y2, width, height, CROP_PADDING_RATIO
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                output_file = os.path.join(output_path, f"{stem}_box{box_index}.jpg")
                cv2.imwrite(output_file, crop)
                crop_count += 1

        print(f"  {class_name}: {crop_count} crops created")
        total += crop_count

    print(f"  total: {total} crops")
    return total


def build_classifier_dataset():
    print_header("Preparing classifier dataset")

    train_output = os.path.join(CLASSIFIER_DATASET_PATH, "train")
    val_output = os.path.join(CLASSIFIER_DATASET_PATH, "val")

    _crop_split(TRAIN_PATH, "Train", train_output)
    _crop_split(VAL_PATH, "Val", val_output)


if __name__ == "__main__":
    build_classifier_dataset()
