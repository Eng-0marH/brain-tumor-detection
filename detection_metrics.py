import os

from config import TUMOR_CLASS_IDS


def compute_iou(box_a, box_b):
    """Intersection over Union of two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_ground_truth_boxes(label_path, image_width, image_height):
    """Read YOLO-format tumor boxes and return them as pixel [x1, y1, x2, y2].

    Non-tumor class ids are skipped, so a No Tumor image yields an empty list
    and is treated as background, matching how the localizer was trained.
    """
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(float(parts[0]))
                x_center, y_center, box_width, box_height = map(float, parts[1:])
            except ValueError:
                continue

            if class_id not in TUMOR_CLASS_IDS:
                continue

            x1 = (x_center - box_width / 2) * image_width
            y1 = (y_center - box_height / 2) * image_height
            x2 = (x_center + box_width / 2) * image_width
            y2 = (y_center + box_height / 2) * image_height

            boxes.append([x1, y1, x2, y2])

    return boxes


def best_iou(predicted_box, ground_truth_boxes):
    """Highest IoU between a prediction and any ground-truth box (0.0 if none)."""
    if predicted_box is None or not ground_truth_boxes:
        return 0.0

    return max(
        compute_iou(predicted_box, gt_box) for gt_box in ground_truth_boxes
    )
