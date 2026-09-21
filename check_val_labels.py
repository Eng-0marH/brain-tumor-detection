import os
import glob

from config import class_from_filename, localization_split_dir


def check_val_labels():
    split_dir = localization_split_dir("Val")
    images_folder = os.path.join(split_dir, "images")
    labels_folder = os.path.join(split_dir, "labels")

    print("\n=== Checking localization validation labels ===")
    print("Labels folder:", labels_folder)

    label_files = sorted(glob.glob(os.path.join(labels_folder, "*.txt")))

    if not label_files:
        print()
        print("No label files found.")
        print("Has localization_prep.build_localization_dataset() been run yet?")
        return 0

    problems = 0
    background_labels = 0
    missing_prefix = 0
    orphan_labels = 0

    for label_path in label_files:
        stem = os.path.splitext(os.path.basename(label_path))[0]

        if class_from_filename(stem) is None:
            print("Missing class prefix:", label_path)
            missing_prefix += 1
            problems += 1

        matching_images = glob.glob(os.path.join(images_folder, stem + ".*"))

        if not matching_images:
            print("Label without image:", label_path)
            orphan_labels += 1
            problems += 1

        with open(label_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            # Empty label = background or No Tumor, which is valid.
            background_labels += 1
            continue

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                print("Invalid label:", label_path)
                problems += 1
                break

            if parts[0] != "0":
                print("Not converted:", label_path, "->", parts[0])
                problems += 1
                break

    print("\nSummary:")
    print("Label files checked:", len(label_files))
    print("Background (empty) labels:", background_labels)
    print("Missing class prefix:", missing_prefix)
    print("Labels without an image:", orphan_labels)
    print("Problems found:", problems)

    return problems


if __name__ == "__main__":
    check_val_labels()
