import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from config import (
    CLASSIFIER_DATASET_PATH, RUNS_PATH, IMAGE_SIZE, CLASSIFIER_BATCH,
    CLASSIFIER_EPOCHS, CLASSIFIER_LR, DEVICE_TRAIN, resolve_device,
)
from reporting import print_header, print_row


def _build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def train_classifier_model():
    print_header("Training EfficientNet-B0 type classifier")

    device = torch.device(resolve_device(DEVICE_TRAIN))
    train_transform, val_transform = _build_transforms()

    train_dataset = datasets.ImageFolder(os.path.join(CLASSIFIER_DATASET_PATH, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(CLASSIFIER_DATASET_PATH, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CLASSIFIER_BATCH, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CLASSIFIER_BATCH, shuffle=False)

    print_row("Training images", len(train_dataset))
    print_row("Validation images", len(val_dataset))
    print_row("Classes", ", ".join(train_dataset.classes))
    print_row("Device", device)
    print()

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CLASSIFIER_LR)

    project_dir = os.path.join(RUNS_PATH, "classification")
    os.makedirs(project_dir, exist_ok=True)
    best_model_path = os.path.join(project_dir, "efficientnet_b0_best.pth")

    best_val_accuracy = 0.0

    for epoch in range(CLASSIFIER_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{CLASSIFIER_EPOCHS}")

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Training Loss: {running_loss / len(train_loader):.4f}")

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved.")

    print()
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    train_classifier_model()
