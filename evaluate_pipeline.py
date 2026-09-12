import os
import cv2
import torch

from config import VAL_PATH, CLASS_NAMES, DEVICE_EVAL
from preprocessing import apply_clahe
from pipeline_inference import (
    build_classifier_transform, load_localization_model, load_classifier_model, run_pipeline,
)


def evaluate_full_pipeline():
    device = torch.device(DEVICE_EVAL)
    transform = build_classifier_transform()

    print("Loading models...")
    loc_model = load_localization_model()
    type_model = load_classifier_model(device)
    print("Models loaded.")

    class_results = {name: {"correct": 0, "total": 0} for name in CLASS_NAMES.values()}
    total_images = 0
    correct_predictions = 0

    for class_name in CLASS_NAMES.values():
        images_path = os.path.join(VAL_PATH, class_name, "images")
        if not os.path.exists(images_path):
            continue

        image_files = [f for f in os.listdir(images_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for image_file in image_files:
            image = cv2.imread(os.path.join(images_path, image_file))
            if image is None:
                continue

            image = apply_clahe(image)

            total_images += 1
            class_results[class_name]["total"] += 1

            result = run_pipeline(image, loc_model, type_model, transform, device)

            if result["predicted_class"] == class_name:
                correct_predictions += 1
                class_results[class_name]["correct"] += 1

        print(f"{class_name}: {class_results[class_name]['correct']}/{class_results[class_name]['total']} correct")

    accuracy = (correct_predictions / total_images) * 100

    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_images})")
    for class_name, results in class_results.items():
        if results["total"] > 0:
            class_accuracy = results["correct"] / results["total"] * 100
            print(f"{class_name}: {class_accuracy:.2f}% ({results['correct']}/{results['total']})")

    return accuracy, class_results


if __name__ == "__main__":
    evaluate_full_pipeline()
