import os
import glob

from config import LOCALIZATION_DATASET_PATH


def check_val_labels():
    labels_folder = os.path.join(LOCALIZATION_DATASET_PATH, "val", "labels")

    label_files = sorted(glob.glob(os.path.join(labels_folder, "*.txt")))

    if not label_files:
        print("No label files found under:", labels_folder)
        print("Has localization_prep.build_localization_dataset() been run yet?")
        return

    problems_found = 0

    for label_path in label_files:
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            # Empty label = background / No Tumor, which is valid.
            continue

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                print("Invalid label:", label_path)
                problems_found += 1
                break

            if parts[0] != "0":
                print("Not converted:", label_path, "->", parts[0])
                problems_found += 1
                break

    print()
    print(f"Checked {len(label_files)} label files under: {labels_folder}")
    print(f"Problems found: {problems_found}")


if __name__ == "__main__":
    check_val_labels()
