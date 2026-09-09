import argparse

from localization_prep import build_localization_dataset
from classifier_prep import build_classifier_dataset
from train_localization import train_localization_model
from train_classifier import train_classifier_model
from evaluate_classifier import evaluate_classifier
from evaluate_pipeline import evaluate_full_pipeline


def run_pipeline(prepare=True, train=True, evaluate=True):
    if prepare:
        print("\n=== Preparing datasets ===")
        build_localization_dataset()
        build_classifier_dataset()

    if train:
        print("\n=== Stage 1: training YOLO11m localizer ===")
        train_localization_model()

        print("\n=== Stage 2: training tumor type classifier ===")
        train_classifier_model()

    if evaluate:
        print("\n=== Evaluating type classifier alone ===")
        evaluate_classifier()

        print("\n=== Evaluating full pipeline ===")
        evaluate_full_pipeline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the brain tumor detection pipeline.")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        prepare=not args.skip_prepare,
        train=not args.skip_train,
        evaluate=not args.skip_evaluate,
    )


"""
python main.py                     # prepare + train + evaluate
python main.py --skip-prepare      # datasets already built
python main.py --skip-train        # models already trained
python main.py --skip-prepare --skip-train   # evaluate only
"""
