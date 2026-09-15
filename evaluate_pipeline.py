import os
import glob
import cv2
import torch

from config import VAL_PATH, CLASS_NAMES, DEVICE_EVAL, resolve_device
from preprocessing import apply_clahe
from pipeline_inference import (
    build_classifier_transform, load_localization_model, load_classifier_model, run_pipeline,
)
from reporting import print_header, print_subheader, print_row


def _get_images(folder):
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(images)


def evaluate_full_pipeline():
    print_header("Full pipeline evaluation")

    device = torch.device(resolve_device(DEVICE_EVAL))
    transform = build_classifier_transform()

    print("Loading models...")
    loc_model = load_localization_model()
    type_model = load_classifier_model(device)
    print("Models loaded.")
    print()

    class_results = {name: {"correct": 0, "total": 0} for name in CLASS_NAMES.values()}
    total_images = 0
    correct_predictions = 0

    for class_name in CLASS_NAMES.values():
        images_path = os.path.join(VAL_PATH, class_name, "images")

        if not os.path.exists(images_path):
            continue

        for image_path in _get_images(images_path):
            image = cv2.imread(image_path)

            if image is None:
                continue

            image = apply_clahe(image)

            total_images += 1
            class_results[class_name]["total"] += 1

            result = run_pipeline(image, loc_model, type_model, transform, device)

            if result["predicted_class"] == class_name:
                correct_predictions += 1
                class_results[class_name]["correct"] += 1

        print(
            f"{class_name}: {class_results[class_name]['correct']}"
            f"/{class_results[class_name]['total']} correct"
        )

    if total_images == 0:
        print()
        print("No validation images found under:", VAL_PATH)
        return 0.0, class_results

    accuracy = correct_predictions / total_images * 100

    print_subheader("Overall")
    print_row("Validation images", total_images)
    print_row("Correct", correct_predictions)
    print_row("Accuracy", f"{accuracy:.2f}%")

    print_subheader("Per-class accuracy")
    for class_name, results in class_results.items():
        if results["total"] > 0:
            class_accuracy = results["correct"] / results["total"] * 100
            print_row(
                class_name,
                f"{class_accuracy:.2f}% ({results['correct']}/{results['total']})",
            )

    print()
    print("Full pipeline evaluation completed.")
    return accuracy, class_results


if __name__ == "__main__":
    evaluate_full_pipeline()
