"""
Reads a predictions CSV (with 'y_true' and 'y_prediction' columns; multi-label values comma-separated)
and saves metrics JSON and confusion matrix figures for:
- PROMISE (single-label): one matrix
- PURE (multi-label): one 2x2 confusion matrix per label, arranged in a grid

Usage (all commands must be run):
  python src/evaluation/evaluate_predictions.py --prediction results/tables/svm_promise_promise_predictions.csv --dataset PROMISE --name svm
  python src/evaluation/evaluate_predictions.py --prediction results/tables/svm_pure_pure_predictions.csv --dataset PURE --name svm
  python src/evaluation/evaluate_predictions.py --prediction results/tables/logreg_promise_promise_predictions.csv --dataset PROMISE --name logreg
  python src/evaluation/evaluate_predictions.py --prediction results/tables/logreg_pure_pure_predictions.csv --dataset PURE --name logreg
  python src/evaluation/evaluate_predictions.py --prediction results/tables/bartmnli_promise_predictions.csv --dataset PROMISE --name bartmnli
  python src/evaluation/evaluate_predictions.py --prediction results/tables/bartmnli_pure_predictions.csv --dataset PURE --name bartmnli
  python src/evaluation/evaluate_predictions.py --prediction results/tables/sbert_promise_predictions.csv --dataset PROMISE --name sbert
  python src/evaluation/evaluate_predictions.py --prediction results/tables/sbert_pure_predictions.csv --dataset PURE --name sbert
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import List

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def is_multilabel_series(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50).tolist()
    return any(("," in value) for value in sample)


def parse_label_lists(values: List[str]) -> List[List[str]]:
    output = []
    for value in values:
        if pd.isna(value):
            output.append([])
        else:
            parts = [t.strip() for t in str(value).split(",")]
            output.append([p for p in parts if p])
    return output


def save_confusion_matrix_single(y_true, y_prediction, classes, out_png, title):
    confusionmatrix = confusion_matrix(y_true, y_prediction, labels=classes)
    figure, axis = plt.subplots(figsize=(6, 5))
    im = axis.imshow(confusionmatrix, interpolation="nearest")
    axis.set_title(title)
    axis.set_xticks(range(len(classes)))
    axis.set_xticklabels(classes, rotation=45, ha="right")
    axis.set_yticks(range(len(classes)))
    axis.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            axis.text(j, i, confusionmatrix[i, j], ha="center", va="center")
    axis.set_ylabel("True")
    axis.set_xlabel("Predicted")
    figure.tight_layout()
    figure.savefig(out_png, dpi=150)
    plt.close(figure)


def save_multilabel_confusions(Y_true: np.ndarray, Y_prediction: np.ndarray, labels: List[str], out_png: str, title: str):
    n_labels = len(labels)
    n_columns = min(3, n_labels)
    n_rows = int(np.ceil(n_labels / n_columns))

    figure, axes = plt.subplots(n_rows, n_columns, figsize=(4 * n_columns, 4 * n_rows))
    if n_rows == 1 and n_columns == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])

    for idx, lab in enumerate(labels):
        row = idx // n_columns
        column = idx % n_columns
        axis = axes[row, column]

        yt = Y_true[:, idx]
        yp = Y_prediction[:, idx]
        confusionmatrix = confusion_matrix(yt, yp, labels=[0, 1])

        im = axis.imshow(confusionmatrix, interpolation="nearest")
        axis.set_title(lab)
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        axis.set_xticklabels(["Neg", "Pos"])
        axis.set_yticklabels(["Neg", "Pos"])
        for i in range(2):
            for j in range(2):
                axis.text(j, i, confusionmatrix[i, j], ha="center", va="center")

        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")

    total_axes = n_rows * n_columns
    for k in range(n_labels, total_axes):
        row = k // n_columns
        column = k % n_columns
        axes[row, column].axis("off")

    figure.suptitle(title)
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.savefig(out_png, dpi=180)
    plt.close(figure)


def main():
    argumentparser = argparse.ArgumentParser()
    argumentparser.add_argument("--prediction", required=True, help="Predictions CSV path")
    argumentparser.add_argument("--dataset", required=True, choices=["PROMISE", "PURE"])
    argumentparser.add_argument("--name", required=True, help="Model name tag for outputs (e.g., svm, logreg, bartmnli, sbert)")
    arguments = argumentparser.parse_args()

    dataframe = pd.read_csv(arguments.prediction)
    if "y_true" not in dataframe.columns or "y_prediction" not in dataframe.columns:
        raise ValueError("CSV must have 'y_true' and 'y_prediction' columns.")

    os.makedirs("results/stats", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    if arguments.dataset == "PROMISE":
        y_true = dataframe["y_true"].astype(str).tolist()
        y_prediction = dataframe["y_prediction"].astype(str).tolist()
        classes = sorted(list(set(y_true) | set(y_prediction)))
        report = classification_report(y_true, y_prediction, zero_division=0, output_dict=True)
        accuracy = accuracy_score(y_true, y_prediction)
        macro_f1 = f1_score(y_true, y_prediction, average="macro", zero_division=0)

        stats_path = f"results/stats/{arguments.name}_promise_metrics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({"accuracy": accuracy, "macro_f1": macro_f1, "report": report}, f, indent=2)
        print(f"Saved metrics to {stats_path}")

        confusionmatrix_path = f"results/figures/{arguments.name}_promise_confusion.png"
        save_confusion_matrix_single(y_true, y_prediction, classes, confusionmatrix_path, f"{arguments.name} - PROMISE")
        print(f"Saved confusion matrix to {confusionmatrix_path}")

    else:
        y_true_lists = parse_label_lists(dataframe["y_true"].tolist())
        y_prediction_lists = parse_label_lists(dataframe["y_prediction"].tolist())

        labels = sorted(list(set(l for row in (y_true_lists + y_prediction_lists) for l in row)))
        idx = {l: i for i, l in enumerate(labels)}

        def binarize(rows):
            Y = np.zeros((len(rows), len(labels)), dtype=int)
            for row, labs in enumerate(rows):
                for l in labs:
                    if l in idx:
                        Y[row, idx[l]] = 1
            return Y

        Y_true = binarize(y_true_lists)
        Y_prediction = binarize(y_prediction_lists)

        report = classification_report(Y_true, Y_prediction, target_names=labels, zero_division=0, output_dict=True)
        macro_f1 = f1_score(Y_true, Y_prediction, average="macro", zero_division=0)
        micro_f1 = f1_score(Y_true, Y_prediction, average="micro", zero_division=0)

        stats_path = f"results/stats/{arguments.name}_pure_metrics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({"macro_f1": macro_f1, "micro_f1": micro_f1, "report": report}, f, indent=2)
        print(f"Saved metrics to {stats_path}")

        confusionmatrix_path = f"results/figures/{arguments.name}_pure_confusion.png"
        save_multilabel_confusions(Y_true, Y_prediction, labels, confusionmatrix_path, f"{arguments.name} - PURE (per-label)")
        print(f"Saved confusion matrices to {confusionmatrix_path}")


if __name__ == "__main__":
    main()
