import os
from ultralytics import YOLO

from config import (
    LOCALIZATION_DATASET_PATH, RUNS_PATH, YOLO_RUN_NAME, YOLO_BASE_MODEL,
    YOLO_EPOCHS, YOLO_IMG_SIZE, YOLO_BATCH, DEVICE_TRAIN,
)


def train_localization_model():
    data_yaml = os.path.join(LOCALIZATION_DATASET_PATH, "data_localization.yaml")

    project_dir = os.path.join(RUNS_PATH, "localization")
    os.makedirs(project_dir, exist_ok=True)

    model = YOLO(YOLO_BASE_MODEL)

    results = model.train(
        data=data_yaml,
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMG_SIZE,
        batch=YOLO_BATCH,
        workers=4,
        project=project_dir,
        name=YOLO_RUN_NAME,
        exist_ok=True,
        seed=42,
        patience=10,
        device=DEVICE_TRAIN,
    )

    print("YOLO11m localization training completed.")
    return results


if __name__ == "__main__":
    train_localization_model()
