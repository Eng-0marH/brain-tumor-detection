import os
import csv
import glob
import cv2
import torch

from config import (
    VAL_PATH, CLASS_NAMES, TUMOR_CLASS_IDS, IOU_THRESHOLD, ANALYSIS_PATH,
    PIPELINE_ERRORS_CSV, PIPELINE_ERROR_SUMMARY_CSV, DEVICE_EVAL, resolve_device,
)
from preprocessing import apply_clahe
from detection_metrics import load_ground_truth_boxes, best_iou
from pipeline_inference import (
    build_classifier_transform, load_localization_model, load_classifier_model, run_pipeline,
)


# Each error is attributed to the stage that caused it, instead of being
# grouped into a single "wrong prediction" bucket.
FALSE_NEGATIVE = "False Negative"
FALSE_POSITIVE = "False Positive"
LOCALIZATION_ERROR = "Localization Error"
CLASSIFICATION_ERROR = "Classification Error"

ERROR_TYPES = (
    FALSE_NEGATIVE,
    FALSE_POSITIVE,
    LOCALIZATION_ERROR,
    CLASSIFICATION_ERROR,
)

CSV_FIELDS = (
    "image",
    "true_class",
    "predicted_class",
    "stage",
    "error_type",
    "detection_confidence",
    "classifier_confidence",
    "iou",
)


def _get_images(folder):
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(images)


def _make_error(image_path, true_class, predicted_class, stage, error_type,
                result, iou):
    return {
        "image": os.path.basename(image_path),
        "image_path": image_path,
        "true_class": true_class,
        "predicted_class": predicted_class,
        "stage": stage,
        "error_type": error_type,
        "detection_confidence": result["detection_confidence"],
        "classifier_confidence": result["classifier_confidence"],
        "iou": iou,
        "detection_box": result["detection_box"],
        "crop_box": result["crop_box"],
    }


def collect_pipeline_errors(write_csv=True):
    """Run the full pipeline over the validation set and categorize every error.

    Returns (errors, stats). Each error carries the boxes needed to draw it, so
    visualize_pipeline_errors can reuse this single pass over the models.
    """
    device = torch.device(resolve_device(DEVICE_EVAL))
    transform = build_classifier_transform()

    print("\n=== Pipeline error analysis ===")
    print("Loading models...")
    loc_model = load_localization_model()
    type_model = load_classifier_model(device)
    print("Models loaded.")

    errors = []

    stats = {
        "total_images": 0,
        "correct": 0,
        "by_error_type": {error_type: 0 for error_type in ERROR_TYPES},
        "by_class": {
            class_name: {"total": 0, "correct": 0}
            for class_name in CLASS_NAMES.values()
        },
    }

    for class_id, true_class in CLASS_NAMES.items():
        images_path = os.path.join(VAL_PATH, true_class, "images")
        labels_path = os.path.join(VAL_PATH, true_class, "labels")

        if not os.path.exists(images_path):
            continue

        is_tumor_class = class_id in TUMOR_CLASS_IDS

        for image_path in _get_images(images_path):
            image = cv2.imread(image_path)

            if image is None:
                continue

            image = apply_clahe(image)
            height, width = image.shape[:2]

            stats["total_images"] += 1
            stats["by_class"][true_class]["total"] += 1

            stem = os.path.splitext(os.path.basename(image_path))[0]
            gt_boxes = load_ground_truth_boxes(
                os.path.join(labels_path, stem + ".txt"), width, height
            )

            result = run_pipeline(image, loc_model, type_model, transform, device)
            iou = best_iou(result["detection_box"], gt_boxes)

            # No Tumor: any detection at all is a localizer false positive
            if not is_tumor_class:
                if result["detected"]:
                    
                    errors.append(_make_error(
                        image_path, true_class, result["predicted_class"],
                        "RF-DETR", FALSE_POSITIVE, result, iou,
                    ))
                    stats["by_error_type"][FALSE_POSITIVE] += 1
                else:
                    stats["correct"] += 1
                    stats["by_class"][true_class]["correct"] += 1
                continue

            # Tumor present but nothing detected
            if not result["detected"]:
                errors.append(_make_error(
                    image_path, true_class, "No Tumor",
                    "RF-DETR", FALSE_NEGATIVE, result, iou,
                ))
                stats["by_error_type"][FALSE_NEGATIVE] += 1
                continue

            # Detected, but the box does not overlap the real tumor
            if gt_boxes and iou < IOU_THRESHOLD:
                errors.append(_make_error(
                    image_path, true_class, result["predicted_class"],
                    "RF-DETR", LOCALIZATION_ERROR, result, iou,
                ))
                stats["by_error_type"][LOCALIZATION_ERROR] += 1
                continue

            # Box was good, so a wrong answer here is the classifier's fault
            if result["predicted_class"] != true_class:
                errors.append(_make_error(
                    image_path, true_class, result["predicted_class"],
                    "EfficientNet", CLASSIFICATION_ERROR, result, iou,
                ))
                stats["by_error_type"][CLASSIFICATION_ERROR] += 1
                continue

            stats["correct"] += 1
            stats["by_class"][true_class]["correct"] += 1

    if write_csv:
        _write_csv(errors, stats)

    _print_report(errors, stats)
    return errors, stats


def _write_csv(errors, stats):
    os.makedirs(ANALYSIS_PATH, exist_ok=True)

    with open(PIPELINE_ERRORS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        writer.writeheader()

        for error in errors:
            writer.writerow({field: error[field] for field in CSV_FIELDS})

    with open(PIPELINE_ERROR_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class"] + list(ERROR_TYPES) + ["total_errors"])

        for class_name in CLASS_NAMES.values():
            counts = [
                sum(
                    1 for error in errors
                    if error["true_class"] == class_name
                    and error["error_type"] == error_type
                )
                for error_type in ERROR_TYPES
            ]
            writer.writerow([class_name] + counts + [sum(counts)])


def _print_report(errors, stats):
    total = stats["total_images"]
    accuracy = (stats["correct"] / total * 100) if total else 0.0

    print("\nOverall:")
    print("Validation images:", total)
    print("Correct:", stats["correct"])
    print("Errors:", len(errors))
    print("Accuracy:", f"{accuracy:.2f}%")

    print("\nErrors by stage:")
    for error_type in ERROR_TYPES:
        print(f"{error_type}: {stats['by_error_type'][error_type]}")

    print("\nErrors by class:")
    for class_name in CLASS_NAMES.values():
        class_errors = [e for e in errors if e["true_class"] == class_name]

        if not class_errors:
            continue

        print(f"{class_name}: {len(class_errors)} errors")
        for error in class_errors:
            detection = (
                f"{error['detection_confidence']:.3f}"
                if error["detection_confidence"] is not None else "-"
            )
            print(
                f"  {error['image']}  {error['error_type']}"
                f"  -> {error['predicted_class']}"
                f"  (det {detection}, IoU {error['iou']:.3f})"
            )

    if errors:
        print("\nSaved files:")
        print("Error details:", PIPELINE_ERRORS_CSV)
        print("Error summary:", PIPELINE_ERROR_SUMMARY_CSV)

    print()
    print("Pipeline error analysis completed.")


if __name__ == "__main__":
    collect_pipeline_errors()
