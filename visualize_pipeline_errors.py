import os
import cv2

from config import VAL_PATH, PIPELINE_ERRORS_PATH
from preprocessing import apply_clahe
from detection_metrics import load_ground_truth_boxes
from analyze_pipeline_errors import collect_pipeline_errors
from reporting import print_header, print_row, print_saved


GROUND_TRUTH_COLOR = (0, 0, 255)     # red
PREDICTION_COLOR = (0, 255, 0)       # green
CROP_COLOR = (255, 200, 0)           # light blue
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_box(image, box, label, color):
    x1, y1, x2, y2 = (int(round(float(value))) for value in box)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image, label, (x1, max(20, y1 - 8)),
        FONT, 0.55, color, 2, cv2.LINE_AA
    )


def _label_lines(error):
    lines = [
        f"Actual: {error['true_class']}",
        f"Predicted: {error['predicted_class']}",
        f"Error: {error['error_type']} ({error['stage']})",
    ]

    if error["detection_confidence"] is not None:
        lines.append(f"RF-DETR confidence: {error['detection_confidence']:.4f}")
    else:
        lines.append("RF-DETR: no detection")

    if error["classifier_confidence"] is not None:
        lines.append(f"Classifier confidence: {error['classifier_confidence']:.4f}")

    lines.append(f"IoU with ground truth: {error['iou']:.4f}")
    return lines


def visualize_pipeline_errors():
    os.makedirs(PIPELINE_ERRORS_PATH, exist_ok=True)

    errors, _ = collect_pipeline_errors()

    print_header("Visualizing pipeline errors")

    saved = 0

    for index, error in enumerate(errors, start=1):
        image = cv2.imread(error["image_path"])

        if image is None:
            continue

        image = apply_clahe(image)
        height, width = image.shape[:2]

        stem = os.path.splitext(error["image"])[0]
        labels_path = os.path.join(VAL_PATH, error["true_class"], "labels")
        gt_boxes = load_ground_truth_boxes(
            os.path.join(labels_path, stem + ".txt"), width, height
        )

        display_image = image.copy()

        for gt_box in gt_boxes:
            _draw_box(display_image, gt_box, "GT", GROUND_TRUTH_COLOR)

        if error["detection_box"] is not None:
            _draw_box(
                display_image, error["detection_box"], "RF-DETR", PREDICTION_COLOR
            )

        if error["crop_box"] is not None:
            _draw_box(display_image, error["crop_box"], "crop", CROP_COLOR)

        y_position = 30
        for line in _label_lines(error):
            cv2.putText(
                display_image, line, (10, y_position),
                FONT, 0.6, PREDICTION_COLOR, 2, cv2.LINE_AA
            )
            y_position += 26

        error_slug = error["error_type"].lower().replace(" ", "_")
        output_file = os.path.join(
            PIPELINE_ERRORS_PATH,
            f"{index:03d}_{error_slug}_{error['true_class']}_{error['image']}",
        )

        cv2.imwrite(output_file, display_image)
        saved += 1

        print(
            f"  {index:03d}  {error['true_class']} -> {error['predicted_class']}"
            f"  ({error['error_type']})"
        )

    print()
    print_row("Error images saved", saved)
    print_saved("Output folder", PIPELINE_ERRORS_PATH)
    print()
    print("Visualization completed.")
    return saved


if __name__ == "__main__":
    visualize_pipeline_errors()
