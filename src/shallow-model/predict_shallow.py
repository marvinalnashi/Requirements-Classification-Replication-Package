"""
Loads a trained shallow model (.pkl) and writes predictions CSVs that match the transformer outputs.
Handles:
- PROMISE-Relabelled (single-label)
- PURE (multi-label)

Usage (all commands must be run):
  python src/shallow-model/predict_shallow.py --model src/shallow-model/results/models/svm_promise.pkl --dataset PROMISE
  python src/shallow-model/predict_shallow.py --model src/shallow-model/results/models/svm_pure.pkl --dataset PURE
  python src/shallow-model/predict_shallow.py --model src/shallow-model/results/models/logreg_promise.pkl --dataset PROMISE
  python src/shallow-model/predict_shallow.py --model src/shallow-model/results/models/logreg_pure.pkl --dataset PURE
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

TEXT_CANDIDATES = ["RequirementText", "requirementtext", "sentence", "text", "requirement", "req_text", "req"]

PROMISE_INDICATORS = ["IsFunctional", "IsQuality", "IsOnlyFunctional", "IsOnlyQuality", "F+Q", "NotR"]

PURE_LABELS_ORDER = ["NFR_boolean", "reliability", "security"]


def find_text_column(dataframe: pd.DataFrame) -> str:
    columns = {c.lower(): c for c in dataframe.columns}
    for candidate in TEXT_CANDIDATES:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    for column in dataframe.columns:
        if any(token in column.lower() for token in ["text", "requirement", "sentence"]):
            return column
    raise ValueError(f"Text column not found. Available columns: {list(dataframe.columns)}")


def derive_promise_labels(dataframe: pd.DataFrame) -> pd.Series:
    dataframe_columns = set(dataframe.columns)
    if set(PROMISE_INDICATORS).issubset(dataframe_columns):
        nfr = (
                (dataframe["IsQuality"] == 1) |
                (dataframe["IsOnlyQuality"] == 1) |
                (dataframe["F+Q"] == 1) |
                (dataframe["NotR"] == 1)
        )
        fr = (
                (dataframe["IsFunctional"] == 1) |
                (dataframe["IsOnlyFunctional"] == 1)
        )
        labels = pd.Series(["FR"] * len(dataframe), index=dataframe.index)
        labels.loc[nfr] = "NFR"
        labels.loc[fr & ~nfr] = "FR"
        return labels

    class_column = next((c for c in ["Class", "class"] if c in dataframe.columns), None)
    if class_column:
        def _map(x):
            s = str(x).lower()
            if "quality" in s or "notr" in s or "f+q" in s or "nfr" in s:
                return "NFR"
            if "functional" in s or "fr" in s:
                return "FR"
            return "FR"
        return dataframe[class_column].map(_map)

    return pd.Series(["FR"] * len(dataframe), index=dataframe.index)


def get_paths(dataset: str):
    here = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(here, "..", "..", "data")
    out_directory = os.path.join(here, "..", "..", "results", "tables")
    os.makedirs(out_directory, exist_ok=True)

    if dataset.upper() == "PROMISE":
        test_csv = os.path.join(data_directory, "PROMISE-Relabelled", "processed", "test.csv")
    elif dataset.upper() == "PURE":
        test_csv = os.path.join(data_directory, "PURE", "processed", "test.csv")
    else:
        raise ValueError("dataset must be PROMISE or PURE")
    return test_csv, out_directory


def main():
    argumentparser = argparse.ArgumentParser()
    argumentparser.add_argument("--model", required=True, help="Path to trained .pkl (svm_*.pkl or logreg_*.pkl)")
    argumentparser.add_argument("--dataset", required=True, choices=["PROMISE", "PURE"], help="Dataset to predict on")
    arguments = argumentparser.parse_args()

    model = joblib.load(arguments.model)
    test_csv, out_directory = get_paths(arguments.dataset)
    dataframe = pd.read_csv(test_csv)

    text_column = find_text_column(dataframe)
    X = dataframe[text_column].astype(str).fillna("")

    if arguments.dataset.upper() == "PROMISE":
        y_true = derive_promise_labels(dataframe).astype(str).tolist()
        output_name = os.path.splitext(os.path.basename(arguments.model))[0].replace(".pkl", "")
        output_csv = os.path.join(out_directory, f"{output_name}_promise_predictions.csv")
        y_prediction = model.predict(X)
        pd.DataFrame({"y_true": y_true, "y_prediction": y_prediction}).to_csv(output_csv, index=False)
        print(f"Saved predictions to {output_csv}")

    else:
        label_column = next((c for c in ["Labels", "labels"] if c in dataframe.columns), None)
        if label_column is None:
            raise ValueError("PURE test.csv missing 'Labels' column.")
        y_true_lists = []
        for x in dataframe[label_column].tolist():
            if pd.isna(x): y_true_lists.append([])
            else:
                parts = [t.strip() for t in str(x).split(",")]
                y_true_lists.append([p for p in parts if p and p.lower() not in {"none", "nan"}])

        Y_prediction = model.predict(X)
        if Y_prediction.ndim == 1:
            Y_prediction = np.vstack([[1 if value == column else 0 for column in PURE_LABELS_ORDER] for value in Y_prediction])

        y_prediction_lists = []
        for row in Y_prediction:
            chosen = [label for label, value in zip(PURE_LABELS_ORDER, row) if int(value) == 1]
            y_prediction_lists.append(chosen)

        output_name = os.path.splitext(os.path.basename(arguments.model))[0].replace(".pkl", "")
        output_csv = os.path.join(out_directory, f"{output_name}_pure_predictions.csv")
        pd.DataFrame({
            "y_true": [",".join(labels) for labels in y_true_lists],
            "y_prediction": [",".join(labels) for labels in y_prediction_lists],
        }).to_csv(output_csv, index=False)
        print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    main()