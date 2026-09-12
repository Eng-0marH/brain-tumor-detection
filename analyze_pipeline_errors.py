import os
import cv2
import torch

from config import VAL_PATH, CLASS_NAMES, DEVICE_EVAL
from preprocessing import apply_clahe
from pipeline_inference import (
    build_classifier_transform, load_localization_model, load_classifier_model, run_pipeline,
)


def analyze_pipeline_errors():
    device = torch.device(DEVICE_EVAL)
    transform = build_classifier_transform()

    print("Loading models...")
    loc_model = load_localization_model()
    type_model = load_classifier_model(device)
    print("Models loaded successfully.")

    errors = []

    for class_name in CLASS_NAMES.values():
        images_path = os.path.join(VAL_PATH, class_name, "images")

        if not os.path.exists(images_path):
            continue

        image_files = [
            f for f in os.listdir(images_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for image_file in image_files:
            image_path = os.path.join(images_path, image_file)
            image = cv2.imread(image_path)

            if image is None:
                continue

            image = apply_clahe(image)

            result = run_pipeline(image, loc_model, type_model, transform, device)

            if result["predicted_class"] != class_name:
                errors.append({
                    "image": image_file,
                    "actual": class_name,
                    "predicted": result["predicted_class"],
                    "detected": result["detected"],
                    "detection_confidence": result["detection_confidence"],
                    "classifier_confidence": result["classifier_confidence"],
                })

    print("\n" + "=" * 70)
    print("Pipeline Error Analysis")
    print("=" * 70)
    print(f"\nTotal errors: {len(errors)}")

    for error in errors:
        print("\nImage:", error["image"])
        print("Actual:", error["actual"])
        print("Predicted:", error["predicted"])
        print("RF-DETR detected:", error["detected"])

        if error["detection_confidence"] is not None:
            print(f"RF-DETR confidence: {error['detection_confidence']:.4f}")

        if error["classifier_confidence"] is not None:
            print(f"Classifier confidence: {error['classifier_confidence']:.4f}")

    print("\n" + "=" * 70)
    print("Error Summary")
    print("=" * 70)

    for actual_class in CLASS_NAMES.values():
        class_errors = [e for e in errors if e["actual"] == actual_class]

        if class_errors:
            print(f"\n{actual_class}: {len(class_errors)} errors")
            for error in class_errors:
                print(f"  {error['image']} -> {error['predicted']}")

    print("\nPipeline error analysis completed!")
    return errors


if __name__ == "__main__":
    analyze_pipeline_errors()
