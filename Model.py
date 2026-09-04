import os
import glob
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yaml
import random
from PIL import Image


# ============================================================
# CLASS NAMES
# ============================================================

Class_Names = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Tumor',
    3: 'Pituitary'
}


# ============================================================
# DATASET PATHS
# ============================================================
# Project folder
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(PROJECT_PATH, "Brain Tumor with Bounding Boxes")

train_path = os.path.join(dataset_path, "Train")

val_path = os.path.join(dataset_path, "Val")


# List files are stored with the dataset
TRAIN_LIST_PATH = os.path.join(dataset_path, 'train_list.txt')
VAL_LIST_PATH = os.path.join(dataset_path, 'val_list.txt')


# ============================================================
# GET LABEL PATH
# ============================================================

def get_label_path(img_path):

    label_dir = os.path.join(
        os.path.dirname(os.path.dirname(img_path)),
        'labels'
    )

    label_path = os.path.join(
        label_dir,
        os.path.splitext(os.path.basename(img_path))[0] + '.txt'
    )

    return label_path


# ============================================================
# VERIFY DATASET
# ============================================================

def verify_dataset(base_path, dataset_name):

    print(f"\n{dataset_name} Dataset Verification")
    print("--------------------------------")

    image_paths = glob.glob(
        os.path.join(base_path, '**', 'images', '*.jpg'),
        recursive=True
    )

    clean_images = []
    corrupt_images = []
    missing_labels = []
    no_tumor_empty_labels = []
    invalid_bboxes = []

    for img_path in image_paths:

        # Check image
        image = cv2.imread(img_path)

        if image is None:
            corrupt_images.append(img_path)
            continue

        # Find corresponding label
        label_path = get_label_path(img_path)

        if not os.path.exists(label_path):
            missing_labels.append(img_path)
            continue

        try:

            with open(label_path, 'r') as f:

                lines = [
                    line.strip()
                    for line in f.readlines()
                    if line.strip()
                ]

        except Exception:

            corrupt_images.append(img_path)
            continue

        # Empty label (No Tumor)
        if len(lines) == 0:

            no_tumor_empty_labels.append(img_path)

            clean_images.append(img_path)

            continue

        valid = True

        for line in lines:

            parts = line.split()

            # YOLO format must contain 5 values
            if len(parts) != 5:

                valid = False
                break

            try:

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:

                valid = False
                break

            # Class must exist
            if class_id not in Class_Names:

                valid = False
                break

            # Coordinates must be normalized
            if not (0 <= x_center <= 1):

                valid = False
                break

            if not (0 <= y_center <= 1):

                valid = False
                break

            # Width and height must be > 0 and <= 1
            if not (0 < width <= 1):

                valid = False
                break

            if not (0 < height <= 1):

                valid = False
                break

        if valid:

            clean_images.append(img_path)

        else:

            invalid_bboxes.append(label_path)

    print(f"Total Scanned:         {len(image_paths)}")
    print(f"Clean Kept:            {len(clean_images)}")
    print(f"  - Annotated Images:  {len(clean_images) - len(no_tumor_empty_labels)}")
    print(f"  - No Tumor (Empty):  {len(no_tumor_empty_labels)}")
    print(f"Corrupt Images:        {len(corrupt_images)}")
    print(f"Missing Labels:        {len(missing_labels)}")
    print(f"Invalid BBoxes:        {len(invalid_bboxes)}")

    if missing_labels:

        print("\nMissing label files:")

        for path in missing_labels[:5]:

            print(path)

    if invalid_bboxes:

        print("\nInvalid bounding-box label files:")

        for path in invalid_bboxes[:5]:

            print(path)

    return clean_images


# ============================================================
# VERIFY TRAINING AND VALIDATION DATA
# ============================================================

def get_verified_datasets():

    train_images = verify_dataset(
        train_path,
        "Train"
    )

    val_images = verify_dataset(
        val_path,
        "Val"
    )

    return train_images, val_images


# ============================================================
# CREATE IMAGE LIST FILES
# ============================================================

def create_txt_files():

    train_images, val_images = get_verified_datasets()

    os.makedirs(dataset_path, exist_ok=True)

    with open(TRAIN_LIST_PATH, 'w') as f:

        for img_path in train_images:

            f.write(img_path + '\n')

    with open(VAL_LIST_PATH, 'w') as f:

        for img_path in val_images:

            f.write(img_path + '\n')

    print("\nList files created successfully.")

    print(f"Training images written:   {len(train_images)}")
    print(f"Validation images written: {len(val_images)}")

    return train_images, val_images


# ============================================================
# GET IMAGE LISTS
# ============================================================

def get_image_lists():

    if (
        os.path.exists(TRAIN_LIST_PATH)
        and os.path.exists(VAL_LIST_PATH)
    ):

        with open(TRAIN_LIST_PATH, 'r') as f:

            train_images = [
                line.strip()
                for line in f
                if line.strip()
            ]

        with open(VAL_LIST_PATH, 'r') as f:

            val_images = [
                line.strip()
                for line in f
                if line.strip()
            ]

        return train_images, val_images

    else:

        return create_txt_files()


# ============================================================
# CREATE YOLO DATA YAML
# ============================================================

def make_yaml():

    yaml_content = {
        'train': TRAIN_LIST_PATH,
        'val': VAL_LIST_PATH,
        'names': Class_Names
    }

    yaml_path = os.path.join(
        dataset_path,
        'data.yaml'
    )

    yaml_text = yaml.safe_dump(
        yaml_content,
        default_flow_style=False,
        sort_keys=False
    )

    with open(
        yaml_path,
        'w',
        encoding='utf-8'
    ) as f:

        f.write(yaml_text)

    print("\ndata.yaml created successfully.")

    return yaml_path


# ============================================================
# DATA DISTRIBUTION
# ============================================================

def data_distributionb():

    train_images, val_images = get_image_lists()

    train_classes = []
    val_classes = []

    for img_path in train_images:

        label_path = get_label_path(img_path)

        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if len(lines) == 0:

            train_classes.append('No Tumor')

        else:

            for line in lines:

                parts = line.split()

                if len(parts) == 5:

                    class_id = int(parts[0])

                    train_classes.append(
                        Class_Names[class_id]
                    )

    for img_path in val_images:

        label_path = get_label_path(img_path)

        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if len(lines) == 0:

            val_classes.append('No Tumor')

        else:

            for line in lines:

                parts = line.split()

                if len(parts) == 5:

                    class_id = int(parts[0])

                    val_classes.append(
                        Class_Names[class_id]
                    )

    train_counts = pd.Series(
        train_classes
    ).value_counts()

    val_counts = pd.Series(
        val_classes
    ).value_counts()

    distribution = pd.DataFrame({
        'Train': train_counts,
        'Validation': val_counts
    }).fillna(0)

    distribution = distribution.reindex(
        Class_Names.values()
    ).fillna(0)

    print("\nClass Distribution:")
    print(distribution)

    # Assign plot to an Axes object
    ax = distribution.plot(
        kind='bar',
        figsize=(10, 6)
    )

    # Add count labels above each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3)

    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Bounding Boxes / Samples")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()

    return distribution


# ============================================================
# TUMOR BOUNDING BOX DIMENSIONS
# ============================================================

def tumor_dimension():

    train_images, val_images = get_image_lists()

    widths = []
    heights = []
    classes = []

    all_images = train_images + val_images

    for img_path in all_images:

        label_path = get_label_path(img_path)

        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                width = float(parts[3])
                height = float(parts[4])

                widths.append(width)
                heights.append(height)

                classes.append(
                    Class_Names[class_id]
                )

    dimension_df = pd.DataFrame({
        'Class': classes,
        'Width': widths,
        'Height': heights
    })

    print("\nTumor Bounding Box Dimensions:")
    print(dimension_df.describe())

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=dimension_df,
        x='Width',
        y='Height',
        hue='Class'
    )

    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)

    plt.title("Tumor Bounding Box Dimensions")
    plt.xlabel("Normalized Width")
    plt.ylabel("Normalized Height")

    plt.tight_layout()
    plt.show()

    return dimension_df


# ============================================================
# IMAGE SHAPES
# ============================================================

def image_shapes():

    train_images, val_images = get_image_lists()

    img_sizes = []

    for img_path in train_images + val_images:

        with Image.open(img_path) as img:
            img_sizes.append(img.size)

    widths, heights = zip(*img_sizes)

    x_min, x_max = np.percentile(widths, [1, 99])
    y_min, y_max = np.percentile(heights, [1, 99])

    plt.figure(figsize=(7, 4))

    plt.hist2d(
        widths,
        heights,
        bins=10,
        range=[[x_min, x_max], [y_min, y_max]],
        cmap='viridis'
    )

    plt.colorbar(label='Image Count')

    plt.title("Image Resolution Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Width (Pixels)", fontsize=10)
    plt.ylabel("Height (Pixels)", fontsize=10)

    plt.xticks(np.linspace(x_min, x_max, 10).astype(int))
    plt.yticks(np.linspace(y_min, y_max, 10).astype(int))

    plt.tight_layout()
    plt.show()

    return widths, heights


# ============================================================
# VISUALIZE SAMPLE IMAGES
# ============================================================

def visualize_samples(num_samples=6):

    train_images, val_images = get_image_lists()

    if len(train_images) == 0:
        print("No training images found to visualize.")
        return

    if len(train_images) < num_samples:

        num_samples = len(train_images)

    samples = random.sample(
        train_images,
        num_samples
    )

    cols = min(num_samples, 3)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 5 * rows)
    )

    if num_samples == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i, img_path in enumerate(samples):

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        label_path = get_label_path(img_path)

        lines = []

        if os.path.exists(label_path):

            with open(label_path, 'r') as f:

                lines = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]

        height, width = image.shape[:2]

        if len(lines) == 0:

            text = "No Tumor"
            font_scale = 0.7
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX

            text_size, _ = cv2.getTextSize(
                text,
                font,
                font_scale,
                thickness
            )

            text_w, text_h = text_size

            # Position at top-right with a 15px padding margin
            text_x = width - text_w - 15
            text_y = text_h + 15

            cv2.putText(
                image,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 255, 0),
                thickness
            )

        else:

            for line in lines:

                parts = line.split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                x1 = int(
                    (x_center - box_width / 2) * width
                )

                y1 = int(
                    (y_center - box_height / 2) * height
                )

                x2 = int(
                    (x_center + box_width / 2) * width
                )

                y2 = int(
                    (y_center + box_height / 2) * height
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    image,
                    Class_Names.get(class_id, "Unknown"),
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

        axes[i].imshow(image)

        axes[i].axis('off')

        axes[i].set_title(
            os.path.basename(img_path)
        )

    for j in range(len(samples), len(axes)):

        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN - DATA PREPARATION + YOLOv8n BASELINE TRAINING
# ============================================================

if __name__ == "__main__":

    
    train_images_paths, val_images_paths = get_image_lists()

   
    yaml_path = make_yaml()

    print("\nDataset preparation completed.")

    print(
        f"Train images: {len(train_images_paths)}"
    )

    print(
        f"Validation images: {len(val_images_paths)}"
    )

    
    data_distributionb()
    tumor_dimension()
    image_shapes()
    visualize_samples(num_samples=9)