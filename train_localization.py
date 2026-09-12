import os
from rfdetr import RFDETRMedium

from config import (
    LOCALIZATION_DATASET_PATH, RFDETR_OUTPUT_DIR, RFDETR_EPOCHS,
    RFDETR_BATCH, RFDETR_GRAD_ACCUM_STEPS, RFDETR_LR, DEVICE_TRAIN,
)


def train_localization_model():
    os.makedirs(RFDETR_OUTPUT_DIR, exist_ok=True)

    
    model = RFDETRMedium()

    results = model.train(
        dataset_dir=LOCALIZATION_DATASET_PATH,
        epochs=RFDETR_EPOCHS,
        batch_size=RFDETR_BATCH,
        grad_accum_steps=RFDETR_GRAD_ACCUM_STEPS,
        lr=RFDETR_LR,
        output_dir=RFDETR_OUTPUT_DIR,
        device=DEVICE_TRAIN,
        early_stopping=True,
        seed=42,
    )

    print("RF-DETR Medium localization training completed.")
    print(f"Best checkpoint: {os.path.join(RFDETR_OUTPUT_DIR, 'checkpoint_best_total.pth')}")
    return results


if __name__ == "__main__":
    train_localization_model()
