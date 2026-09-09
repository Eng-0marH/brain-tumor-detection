import os
import cv2
import torch
from torch import nn
from torchvision import transforms, models
from ultralytics import YOLO

from config import (
    VAL_PATH, CLASS_NAMES, YOLO_MODEL_PATH, CLASSIFIER_MODEL_PATH,
    IMAGE_SIZE, CONFIDENCE_THRESHOLD, CROP_PADDING_RATIO, DEVICE_EVAL, TUMOR_CLASS_NAMES,
)
from preprocessing import apply_clahe


def _build_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _load_type_model(device):
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(TUMOR_CLASS_NAMES))
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def _pad_and_clip(x1, y1, x2, y2, width, height, padding_ratio):
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    x1 = max(0, min(x1 - pad_x, width))
    y1 = max(0, min(y1 - pad_y, height))
    x2 = max(0, min(x2 + pad_x, width))
    y2 = max(0, min(y2 + pad_y, height))

    return x1, y1, x2, y2


def _predict(image, yolo_model, type_model, transform, device):
    """
    Returns the predicted tumor type for one image, or "No Tumor" if
    YOLO finds nothing above CONFIDENCE_THRESHOLD. Only the single
    highest-confidence box is cropped and classified.
    """
    results = yolo_model.predict(source=image, conf=CONFIDENCE_THRESHOLD, device=DEVICE_EVAL, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return "No Tumor"

    height, width = image.shape[:2]

    best_box_index = torch.argmax(boxes.conf).item()
    box = boxes.xyxy[best_box_index].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)

    x1, y1, x2, y2 = _pad_and_clip(x1, y1, x2, y2, width, height, CROP_PADDING_RATIO)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return "No Tumor"

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    input_tensor = transform(crop_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = type_model(input_tensor)
        _, prediction = torch.max(outputs, 1)

    return TUMOR_CLASS_NAMES[prediction.item()]


def evaluate_full_pipeline():
    device = torch.device(DEVICE_EVAL)
    transform = _build_transform()

    print("Loading models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    type_model = _load_type_model(device)
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

            predicted_class = _predict(image, yolo_model, type_model, transform, device)

            if predicted_class == class_name:
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
