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

CLASS_PREFIXES = {
    name: name.lower().replace(" ", "_") + "_"
    for name in CLASS_NAMES.values()
}

# built fresh into build/, source is untouched 
BUILD_PATH = os.path.join(PROJECT_PATH, "build")

LOCALIZATION_DATASET_PATH = os.path.join(BUILD_PATH, "localization_dataset")
CLASSIFIER_DATASET_PATH = os.path.join(BUILD_PATH, "classifier_dataset")

# RF-DETR expects its splits to be named train / valid / test, not train / val.
LOCALIZATION_SPLIT_DIRS = {
    "Train": "train",
    "Val": "valid",
}

# RF-DETR reads YOLO-format labels directly when this is passed to model.train()
LOCALIZATION_DATASET_FORMAT = "yolo"

# trained model locations
RUNS_PATH = os.path.join(PROJECT_PATH, "runs")
RFDETR_RUN_NAME = "rfdetr_medium_tumor_localization"
RFDETR_OUTPUT_DIR = os.path.join(RUNS_PATH, "localization", RFDETR_RUN_NAME)

# RF-DETR writes several checkpoints. The EMA weights are usually the stronger
# ones, so they are preferred and the plain best checkpoint is the fallback.
RFDETR_CHECKPOINT_NAMES = (
    "checkpoint_best_ema.pth",
    "checkpoint_best_total.pth",
)

CLASSIFIER_MODEL_PATH = os.path.join(
    RUNS_PATH, "classification", "efficientnet_b0_best.pth"
)

# hyperparameters
IMAGE_SIZE = 224
CROP_PADDING_RATIO = 0.2
CONFIDENCE_THRESHOLD = 0.25

# a detection counts as correctly localized above this IoU with a ground-truth box
IOU_THRESHOLD = 0.5

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
PIPELINE_ERRORS_CSV = os.path.join(ANALYSIS_PATH, "pipeline_errors.csv")
PIPELINE_ERROR_SUMMARY_CSV = os.path.join(ANALYSIS_PATH, "pipeline_error_summary.csv")


def class_prefix(class_name):
    """Filename prefix used for a class inside the flattened localization set."""
    return CLASS_PREFIXES[class_name]


def class_from_filename(filename):
    """Recover the source class of a flattened localization image, or None."""
    lowered = os.path.basename(filename).lower()

    for class_name, prefix in CLASS_PREFIXES.items():
        if lowered.startswith(prefix):
            return class_name

    return None


def localization_split_dir(split_name):
    
    return os.path.join(
        LOCALIZATION_DATASET_PATH, LOCALIZATION_SPLIT_DIRS[split_name]
    )


def resolve_rfdetr_checkpoint():
    
    for name in RFDETR_CHECKPOINT_NAMES:
        path = os.path.join(RFDETR_OUTPUT_DIR, name)

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No RF-DETR checkpoint found in: " + RFDETR_OUTPUT_DIR
    )


def resolve_device(preferred):
    
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"

    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"

    if preferred == "cpu":
        return "cpu"

    print(f"Device '{preferred}' is unavailable, falling back to CPU.")
    return "cpu"
