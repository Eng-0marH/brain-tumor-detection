import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    CLASSIFIER_DATASET_PATH, CLASSIFIER_MODEL_PATH, IMAGE_SIZE,
    CLASSIFIER_BATCH, DEVICE_EVAL, resolve_device,
)
from reporting import print_header, print_subheader, print_row


def evaluate_classifier():
    print_header("Type classifier evaluation")

    device = torch.device(resolve_device(DEVICE_EVAL))

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = datasets.ImageFolder(
        os.path.join(CLASSIFIER_DATASET_PATH, "val"), transform=val_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=CLASSIFIER_BATCH, shuffle=False)

    print_row("Validation crops", len(val_dataset))
    print_row("Classes", ", ".join(val_dataset.classes))
    print_row("Device", device)

    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(val_dataset.classes))
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    if not all_labels:
        print()
        print("No validation crops found under:", CLASSIFIER_DATASET_PATH)
        return 0.0

    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels) * 100

    print_subheader("Overall")
    print_row("Accuracy", f"{accuracy:.2f}%")

    print_subheader("Classification report")
    print(classification_report(
        all_labels, all_predictions, target_names=val_dataset.classes, digits=4
    ))

    print_subheader("Confusion matrix")
    print(confusion_matrix(all_labels, all_predictions))

    print()
    print("Type classifier evaluation completed.")
    return accuracy


if __name__ == "__main__":
    evaluate_classifier()
