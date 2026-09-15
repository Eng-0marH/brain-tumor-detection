import cv2
import torch
from torch import nn
from torchvision import transforms, models
from rfdetr import RFDETRMedium

from config import (
    CLASSIFIER_MODEL_PATH, IMAGE_SIZE, CONFIDENCE_THRESHOLD,
    CROP_PADDING_RATIO, TUMOR_CLASS_NAMES, resolve_rfdetr_checkpoint,
)
from preprocessing import pad_and_clip


def build_classifier_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_localization_model():
    checkpoint_path = resolve_rfdetr_checkpoint()
    print(f"RF-DETR checkpoint: {checkpoint_path}")
    return RFDETRMedium(pretrain_weights=checkpoint_path)


def load_classifier_model(device):
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(TUMOR_CLASS_NAMES))
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def run_pipeline(image_bgr, loc_model, type_model, transform, device):
    """Detect a tumor, then classify its type.

    Returns:
        predicted_class        final 4-class answer ("No Tumor" if nothing detected)
        detected               whether RF-DETR produced any box
        detection_confidence   confidence of the selected box
        classifier_confidence  softmax confidence of the type prediction
        detection_box          selected box as RF-DETR returned it, used for IoU
        crop_box               the padded box actually fed to the classifier
    """
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detections = loc_model.predict(image_rgb, threshold=CONFIDENCE_THRESHOLD)

    result = {
        "predicted_class": "No Tumor",
        "detected": len(detections.xyxy) > 0,
        "detection_confidence": None,
        "classifier_confidence": None,
        "detection_box": None,
        "crop_box": None,
    }

    if not result["detected"]:
        return result

    best_index = int(detections.confidence.argmax())
    result["detection_confidence"] = float(detections.confidence[best_index])

    x1, y1, x2, y2 = (int(value) for value in detections.xyxy[best_index])
    result["detection_box"] = (x1, y1, x2, y2)

    # padded box gives the classifier the same framing it was trained on
    x1, y1, x2, y2 = pad_and_clip(x1, y1, x2, y2, width, height, CROP_PADDING_RATIO)
    result["crop_box"] = (x1, y1, x2, y2)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return result

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    input_tensor = transform(crop_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = type_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    result["predicted_class"] = TUMOR_CLASS_NAMES[prediction.item()]
    result["classifier_confidence"] = confidence.item()
    return result
