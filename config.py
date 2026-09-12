import os

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))


DATASET_PATH = os.path.join(PROJECT_PATH, "Brain Tumor with Bounding Boxes")
TRAIN_PATH = os.path.join(DATASET_PATH, "Train")
VAL_PATH = os.path.join(DATASET_PATH, "Val")


CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary",
}

TUMOR_CLASS_IDS = {0, 1, 3}

# Output order for the type classifier (Glioma / Meningioma / Pituitary)
TUMOR_CLASS_NAMES = [CLASS_NAMES[i] for i in sorted(TUMOR_CLASS_IDS)]

# built fresh into build/, source is untouched 
BUILD_PATH = os.path.join(PROJECT_PATH, "build")

LOCALIZATION_DATASET_PATH = os.path.join(BUILD_PATH, "localization_dataset")
CLASSIFIER_DATASET_PATH = os.path.join(BUILD_PATH, "classifier_dataset")

#trained model locations
RUNS_PATH = os.path.join(PROJECT_PATH, "runs")
RFDETR_RUN_NAME = "rfdetr_medium_tumor_localization"
RFDETR_OUTPUT_DIR = os.path.join(RUNS_PATH, "localization", RFDETR_RUN_NAME)
RFDETR_MODEL_PATH = os.path.join(RFDETR_OUTPUT_DIR, "checkpoint_best_total.pth")

CLASSIFIER_MODEL_PATH = os.path.join(
    RUNS_PATH, "classification", "efficientnet_b0_best.pth"
)

# hyperparameters
IMAGE_SIZE = 224
CROP_PADDING_RATIO = 0.2
CONFIDENCE_THRESHOLD = 0.25

# CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

RFDETR_EPOCHS = 50
RFDETR_BATCH = 4
RFDETR_GRAD_ACCUM_STEPS = 4
RFDETR_LR = 1e-4

CLASSIFIER_EPOCHS = 50
CLASSIFIER_BATCH = 64
CLASSIFIER_LR = 0.0001

DEVICE_TRAIN = "cuda"
DEVICE_EVAL = "cuda"

# analysis / error outputs 
ANALYSIS_PATH = os.path.join(PROJECT_PATH, "analysis")
PIPELINE_ERRORS_PATH = os.path.join(PROJECT_PATH, "pipeline_errors")
