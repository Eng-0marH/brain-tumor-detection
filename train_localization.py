import os
from rfdetr import RFDETRMedium

from config import (
    LOCALIZATION_DATASET_PATH, LOCALIZATION_DATASET_FORMAT, RFDETR_OUTPUT_DIR,
    RFDETR_EPOCHS, RFDETR_BATCH, RFDETR_GRAD_ACCUM_STEPS, RFDETR_LR,
    DEVICE_TRAIN, resolve_device,
)


def train_localization_model():
    print("\n=== Training RF-DETR Medium localizer ===")

    os.makedirs(RFDETR_OUTPUT_DIR, exist_ok=True)

    device = resolve_device(DEVICE_TRAIN)

    print(f"Dataset:  {LOCALIZATION_DATASET_PATH}")
    print(f"Format:   {LOCALIZATION_DATASET_FORMAT}")
    print(f"Device:   {device}")
    print(f"Epochs:   {RFDETR_EPOCHS}")
    print()

    model = RFDETRMedium()

    results = model.train(
        dataset_dir=LOCALIZATION_DATASET_PATH,
        dataset_file=LOCALIZATION_DATASET_FORMAT,
        epochs=RFDETR_EPOCHS,
        batch_size=RFDETR_BATCH,
        grad_accum_steps=RFDETR_GRAD_ACCUM_STEPS,
        lr=RFDETR_LR,
        output_dir=RFDETR_OUTPUT_DIR,
        device=device,
        early_stopping=True,
        seed=42,
    )

    print()
    print("RF-DETR Medium localization training completed.")
    print("Checkpoints written to:", RFDETR_OUTPUT_DIR)
    return results


if __name__ == "__main__":
    train_localization_model()
