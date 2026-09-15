import argparse

from localization_prep import build_localization_dataset
from classifier_prep import build_classifier_dataset
from check_val_labels import check_val_labels
from train_localization import train_localization_model
from train_classifier import train_classifier_model
from evaluate_classifier import evaluate_classifier
from evaluate_pipeline import evaluate_full_pipeline
from analyze_pipeline_errors import collect_pipeline_errors
from visualize_pipeline_errors import visualize_pipeline_errors


def run_pipeline(prepare=True, train=True, evaluate=True, analyze=True, visualize=False):
    if prepare:
        build_localization_dataset()
        build_classifier_dataset()
        check_val_labels()

    if train:
        train_localization_model()
        train_classifier_model()

    if evaluate:
        evaluate_classifier()
        evaluate_full_pipeline()

    if visualize:
        # runs the models once and also writes the error CSVs
        visualize_pipeline_errors()
    elif analyze:
        collect_pipeline_errors()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the brain tumor detection pipeline.")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--visualize", action="store_true",
                        help="also save an annotated image for every error")
    args = parser.parse_args()

    run_pipeline(
        prepare=not args.skip_prepare,
        train=not args.skip_train,
        evaluate=not args.skip_evaluate,
        analyze=not args.skip_analyze,
        visualize=args.visualize,
    )


"""
python main.py                                  # prepare + train + evaluate + analyze
python main.py --skip-prepare                   # datasets already built
python main.py --skip-train                     # models already trained
python main.py --skip-prepare --skip-train      # evaluate + analyze only
python main.py --skip-prepare --skip-train --visualize   # also save error images
"""
