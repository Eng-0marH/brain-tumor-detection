import os
import cv2
import torch

from config import VAL_PATH, CLASS_NAMES, DEVICE_EVAL, PIPELINE_ERRORS_PATH
from preprocessing import apply_clahe
from pipeline_inference import (
    build_classifier_transform, load_localization_model, load_classifier_model, run_pipeline,
)


def visualize_pipeline_errors():
    os.makedirs(PIPELINE_ERRORS_PATH, exist_ok=True)

    device = torch.device(DEVICE_EVAL)
    transform = build_classifier_transform()

    print("Loading models...")
    loc_model = load_localization_model()
    type_model = load_classifier_model(device)
    print("Models loaded successfully.")

    error_count = 0

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

            if result["predicted_class"] == class_name:
                continue

            error_count += 1
            display_image = image.copy()

            if result["box"] is not None:
                x1, y1, x2, y2 = result["box"]
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            lines = [f"Actual: {class_name}", f"Predicted: {result['predicted_class']}"]

            if result["detection_confidence"] is not None:
                lines.append(f"RF-DETR confidence: {result['detection_confidence']:.4f}")
            else:
                lines.append("RF-DETR: No detection")

            if result["classifier_confidence"] is not None:
                lines.append(f"Classifier confidence: {result['classifier_confidence']:.4f}")

            y_position = 30
            for line in lines:
                cv2.putText(
                    display_image, line, (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                )
                y_position += 30

            output_file = os.path.join(
                PIPELINE_ERRORS_PATH, f"error_{error_count:02d}_{class_name}_{image_file}"
            )
            cv2.imwrite(output_file, display_image)

            print(f"Saved {error_count}: {class_name} -> {result['predicted_class']}")

    print("\n" + "=" * 60)
    print("Visualization completed")
    print("=" * 60)
    print(f"Total error images saved: {error_count}")
    print(f"Output folder: {PIPELINE_ERRORS_PATH}")


if __name__ == "__main__":
    visualize_pipeline_errors()
