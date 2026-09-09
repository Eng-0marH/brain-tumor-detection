import os
import cv2
import torch
from torch import nn
from torchvision import transforms, models
from ultralytics import YOLO

from config import (
    VAL_PATH, CLASS_NAMES, YOLO_MODEL_PATH, CLASSIFIER_MODEL_PATH,
    IMAGE_SIZE, CONFIDENCE_THRESHOLD, CROP_PADDING_RATIO, DEVICE_EVAL,
    TUMOR_CLASS_NAMES, PIPELINE_ERRORS_PATH,
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


def visualize_pipeline_errors():
    os.makedirs(PIPELINE_ERRORS_PATH, exist_ok=True)

    device = torch.device(DEVICE_EVAL)
    transform = _build_transform()

    print("Loading models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    type_model = _load_type_model(device)
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

            results = yolo_model.predict(
                source=image, conf=CONFIDENCE_THRESHOLD, device=DEVICE_EVAL, verbose=False
            )
            boxes = results[0].boxes

            yolo_detected = boxes is not None and len(boxes) > 0
            predicted_class = "No Tumor"
            yolo_confidence = None
            classifier_confidence = None
            selected_box = None

            if yolo_detected:
                best_box_index = torch.argmax(boxes.conf).item()
                yolo_confidence = boxes.conf[best_box_index].item()
                selected_box = boxes.xyxy[best_box_index].cpu().numpy()

                x1, y1, x2, y2 = map(int, selected_box)
                height, width = image.shape[:2]

                box_w, box_h = x2 - x1, y2 - y1
                pad_x = int(box_w * CROP_PADDING_RATIO)
                pad_y = int(box_h * CROP_PADDING_RATIO)

                x1 = max(0, min(x1 - pad_x, width))
                y1 = max(0, min(y1 - pad_y, height))
                x2 = max(0, min(x2 + pad_x, width))
                y2 = max(0, min(y2 + pad_y, height))

                crop = image[y1:y2, x1:x2]

                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(crop_rgb).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = type_model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, prediction = torch.max(probabilities, 1)

                    predicted_class = TUMOR_CLASS_NAMES[prediction.item()]
                    classifier_confidence = confidence.item()

            if predicted_class == class_name:
                continue

            error_count += 1
            display_image = image.copy()

            if selected_box is not None:
                x1, y1, x2, y2 = map(int, selected_box)
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            lines = [f"Actual: {class_name}", f"Predicted: {predicted_class}"]

            if yolo_confidence is not None:
                lines.append(f"YOLO confidence: {yolo_confidence:.4f}")
            else:
                lines.append("YOLO: No detection")

            if classifier_confidence is not None:
                lines.append(f"Classifier confidence: {classifier_confidence:.4f}")

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

            print(f"Saved {error_count}: {class_name} -> {predicted_class}")

    print("\n" + "=" * 60)
    print("Visualization completed")
    print("=" * 60)
    print(f"Total error images saved: {error_count}")
    print(f"Output folder: {PIPELINE_ERRORS_PATH}")


if __name__ == "__main__":
    visualize_pipeline_errors()
