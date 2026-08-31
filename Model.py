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


Class_Names = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Tumor',
    3: 'Pituitary'
}


dataset_path = r"C:\Users\omar\Desktop\MLA Project\Brain Tumor with Bounding Boxes"

train_path = r"C:\Users\omar\Desktop\MLA Project\Brain Tumor with Bounding Boxes\Train"

val_path = r"C:\Users\omar\Desktop\MLA Project\Brain Tumor with Bounding Boxes\Val"


TRAIN_LIST_PATH = os.path.abspath('train_list.txt')
VAL_LIST_PATH = os.path.abspath('val_list.txt')


def verify_dataset(base_path, split_name):

    image_paths = glob.glob(
        os.path.join(base_path, '**', 'images', '*.jpg'),
        recursive=True
    )

    clean_images_paths = []
    corrupt_images = []
    missing_labels = []
    empty_labels = []
    invalid_boxes = []

    for img_path in image_paths:

        cv_img = cv2.imread(img_path)

        if cv_img is None:
            corrupt_images.append(img_path)
            continue

        label_dir = os.path.join(
            os.path.dirname(os.path.dirname(img_path)),
            'labels'
        )

        label_path = os.path.join(
            label_dir,
            os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        )

        if not os.path.exists(label_path):
            missing_labels.append(img_path)
            continue

        with open(label_path, 'r') as f:
            lines = [
                l.strip()
                for l in f.readlines()
                if l.strip()
            ]

        if not lines:
            empty_labels.append(label_path)
            continue

        has_invalid_box = False

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                invalid_boxes.append(
                    (label_path, "Wrong column count")
                )
                has_invalid_box = True
                break

            try:

                cls_id, x, y, w, h = map(float, parts)

                if int(cls_id) not in Class_Names.keys():

                    invalid_boxes.append(
                        (
                            label_path,
                            f"Unknown Class ID: {cls_id}"
                        )
                    )

                    has_invalid_box = True
                    break

                if not (
                    0.0 <= x <= 1.0
                    and 0.0 <= y <= 1.0
                    and 0.0 < w <= 1.0
                    and 0.0 < h <= 1.0
                ):

                    invalid_boxes.append(
                        (
                            label_path,
                            f"Coordinates out of bounds: "
                            f"x={x}, y={y}, "
                            f"w={w}, h={h}"
                        )
                    )

                    has_invalid_box = True
                    break

            except ValueError:

                invalid_boxes.append(
                    (
                        label_path,
                        "Non-numeric values in label"
                    )
                )

                has_invalid_box = True
                break

        if not has_invalid_box:
            clean_images_paths.append(img_path)

    print(f"\n{split_name} Dataset Verification")
    print("--------------------------------")
    print(f"Total Scanned:  {len(image_paths)}")
    print(f"Clean Kept:     {len(clean_images_paths)}")
    print(f"Corrupt Images: {len(corrupt_images)}")
    print(f"Missing Labels: {len(missing_labels)}")
    print(f"Empty Labels:   {len(empty_labels)}")
    print(f"Invalid BBoxes: {len(invalid_boxes)}")

    return clean_images_paths


def create_txt_files():

    train_images_paths = verify_dataset(
        train_path,
        "Train"
    )

    val_images_paths = verify_dataset(
        val_path,
        "Val"
    )

    with open(TRAIN_LIST_PATH, 'w') as f:
        f.write('\n'.join(train_images_paths))

    with open(VAL_LIST_PATH, 'w') as f:
        f.write('\n'.join(val_images_paths))

    return train_images_paths, val_images_paths


def read_txt_files():

    with open(TRAIN_LIST_PATH, 'r') as f:

        train_images_paths = [
            line.strip()
            for line in f
            if line.strip()
        ]

    with open(VAL_LIST_PATH, 'r') as f:

        val_images_paths = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return train_images_paths, val_images_paths


def get_image_lists():

    global train_images_paths
    global val_images_paths

    if (
        os.path.exists(TRAIN_LIST_PATH)
        and os.path.exists(VAL_LIST_PATH)
    ):

        train_images_paths, val_images_paths = (
            read_txt_files()
        )

    else:

        train_images_paths, val_images_paths = (
            create_txt_files()
        )

    return train_images_paths, val_images_paths


train_images_paths, val_images_paths = get_image_lists()


def count_by_class(list_txt_path, base_path):

    with open(list_txt_path, 'r') as f:

        image_paths = [
            line.strip()
            for line in f
            if line.strip()
        ]

    counts = []

    for cls in Class_Names.values():

        class_folder = os.path.normpath(
            os.path.join(
                base_path,
                cls,
                'images'
            )
        )

        count = sum(
            1
            for p in image_paths
            if os.path.normpath(p).startswith(class_folder)
        )

        counts.append(count)

    return counts


train_counts = count_by_class(
    TRAIN_LIST_PATH,
    train_path
)

val_counts = count_by_class(
    VAL_LIST_PATH,
    val_path
)


def make_yaml():

    yaml_content = {
        'train': TRAIN_LIST_PATH,
        'val': VAL_LIST_PATH,
        'names': Class_Names
    }

    with open('data.yaml', 'w') as f:

        yaml.dump(
            yaml_content,
            f,
            default_flow_style=False
        )


def data_distributionb():

    classes = list(Class_Names.values())
    x = np.arange(len(classes))

    plt.figure(figsize=(9, 5))

    bars_train = plt.bar(
        x - 0.2,
        train_counts,
        0.4,
        label='Train',
        color='#4C72B0'
    )

    bars_val = plt.bar(
        x + 0.2,
        val_counts,
        0.4,
        label='Validation',
        color='#DD8452'
    )

    plt.title(
        "Data Distribution: Train & Val",
        fontsize=14,
        fontweight="bold"
    )

    plt.xticks(x, classes)
    plt.legend()

    for bar in bars_train:

        yval = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 15,
            str(yval),
            ha='center',
            fontweight='bold'
        )

    for bar in bars_val:

        yval = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 15,
            str(yval),
            ha='center',
            fontweight='bold'
        )

    plt.tight_layout()
    plt.show()


def tumor_dimension():

    box_widths = []
    box_heights = []

    for img_path in train_images_paths + val_images_paths:

        label_dir = os.path.join(
            os.path.dirname(os.path.dirname(img_path)),
            'labels'
        )

        txt_path = os.path.join(
            label_dir,
            os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        )

        if os.path.exists(txt_path):

            with open(txt_path, 'r') as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) == 5:

                        box_widths.append(
                            float(parts[3])
                        )

                        box_heights.append(
                            float(parts[4])
                        )

    plt.figure(figsize=(7, 5))

    plt.scatter(
        box_widths,
        box_heights,
        alpha=0.3,
        color='purple'
    )

    plt.title(
        "Tumor Dimensions (Width & Height)",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Width", fontsize=12)
    plt.ylabel("Height", fontsize=12)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()


def image_shapes():

    img_sizes = []

    for img_path in np.array(
        train_images_paths + val_images_paths
    ):

        with Image.open(img_path) as img:
            img_sizes.append(img.size)

    widths, heights = zip(*img_sizes)

    x_min, x_max = np.percentile(
        widths,
        [1, 99]
    )

    y_min, y_max = np.percentile(
        heights,
        [1, 99]
    )

    plt.figure(figsize=(7, 4))

    plt.hist2d(
        widths,
        heights,
        bins=10,
        range=[
            [x_min, x_max],
            [y_min, y_max]
        ],
        cmap='viridis'
    )

    plt.colorbar(
        label='Image Count'
    )

    plt.title(
        "Image Resolution Distribution",
        fontsize=12,
        fontweight="bold"
    )

    plt.xlabel(
        "Width (Pixels)",
        fontsize=10
    )

    plt.ylabel(
        "Height (Pixels)",
        fontsize=10
    )

    plt.xticks(
        np.linspace(
            x_min,
            x_max,
            10
        ).astype(int)
    )

    plt.yticks(
        np.linspace(
            y_min,
            y_max,
            10
        ).astype(int)
    )

    plt.tight_layout()
    plt.show()


def visualize_samples(image_list, num_samples):

    sample_paths = random.sample(
        image_list,
        num_samples
    )

    fig, axes = plt.subplots(
        1,
        num_samples,
        figsize=(16, 4)
    )

    if num_samples == 1:
        axes = [axes]

    for ax, img_path in zip(
        axes,
        sample_paths
    ):

        img = cv2.imread(img_path)

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        h_img, w_img, _ = img.shape

        label_dir = os.path.join(
            os.path.dirname(os.path.dirname(img_path)),
            'labels'
        )

        txt_path = os.path.join(
            label_dir,
            os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        )

        if os.path.exists(txt_path):

            with open(txt_path, 'r') as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) == 5:

                        cls_id, x, y, w, h = map(
                            float,
                            parts
                        )

                        x1 = int(
                            (x - w / 2) * w_img
                        )

                        y1 = int(
                            (y - h / 2) * h_img
                        )

                        x2 = int(
                            (x + w / 2) * w_img
                        )

                        y2 = int(
                            (y + h / 2) * h_img
                        )

                        label_name = Class_Names.get(
                            int(cls_id),
                            'Unknown'
                        )

                        cv2.rectangle(
                            img,
                            (x1, y1),
                            (x2, y2),
                            (255, 0, 0),
                            2
                        )

                        cv2.putText(
                            img,
                            label_name,
                            (
                                x1,
                                max(y1 - 5, 15)
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2
                        )

        ax.imshow(img)
        ax.axis('off')

        ax.set_title(
            os.path.basename(img_path),
            fontsize=10
        )

    plt.tight_layout()
    plt.show()


make_yaml()


model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    workers=4,
    name='yolov8n_baseline',
    seed=42,
    patience=10,
)