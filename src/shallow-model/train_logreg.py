"""
Trains a Logistic Regression classifier for requirements classification.
- Handles PROMISE-Relabelled (single-label) robustly and derives labels if needed.
- Handles PURE (multi-label) using comma-separated labels.
"""

import os
import sys
from typing import Tuple, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, f1_score


TEXT_CANDIDATES = [
    "RequirementText", "requirementtext", "sentence", "text", "requirement", "req_text", "req"
]
LABEL_CANDIDATES = [
    "Labels", "labels", "Label", "label", "Class", "class"
]

PROMISE_INDICATORS = ["IsFunctional", "IsQuality", "IsOnlyFunctional", "IsOnlyQuality", "F+Q", "NotR"]


def find_column(dataframe: pd.DataFrame, candidates: List[str]) -> str:
    columns = {c.lower(): c for c in dataframe.columns}
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    for column in dataframe.columns:
        if any(token in column.lower() for token in ["text", "requirement", "sentence"]):
            return column
    raise ValueError(f"Could not find a suitable column among: {candidates}. Available: {list(dataframe.columns)}")


def derive_promise_binary_labels(dataframe: pd.DataFrame) -> pd.Series:
    dataframe_columns = set(dataframe.columns)

    if set(PROMISE_INDICATORS).issubset(dataframe_columns):
        nfr_mask = (
                (dataframe["IsQuality"] == 1) |
                (dataframe["IsOnlyQuality"] == 1) |
                (dataframe["F+Q"] == 1) |
                (dataframe["NotR"] == 1)
        )
        fr_mask = (
                (dataframe["IsFunctional"] == 1) |
                (dataframe["IsOnlyFunctional"] == 1)
        )
        labels = pd.Series(["FR"] * len(dataframe), index=dataframe.index)
        labels.loc[nfr_mask] = "NFR"
        labels.loc[fr_mask & ~nfr_mask] = "FR"
        return labels

    class_column = None
    for candidate in ["Class", "class"]:
        if candidate in dataframe.columns:
            class_column = candidate
            break

    if class_column is not None:
        def _map_class(value):
            s = str(value).lower()
            if "quality" in s or "notr" in s or "f+q" in s or "nfr" in s:
                return "NFR"
            if "functional" in s or "fr" in s:
                return "FR"
            return "FR"
        return dataframe[class_column].map(_map_class)

    return pd.Series(["FR"] * len(dataframe), index=dataframe.index)


def is_multilabel_series(series: pd.Series) -> bool:
    if series.empty:
        return False
    sample = series.dropna().astype(str).head(20).tolist()
    return any(("," in value or ";" in value) for value in sample)


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, bool]:
    train_dataframe = pd.read_csv(train_path)
    test_dataframe  = pd.read_csv(test_path)

    text_column = find_column(train_dataframe, TEXT_CANDIDATES)
    if text_column not in test_dataframe.columns:
        if text_column.lower() in [c.lower() for c in test_dataframe.columns]:
            text_column = [c for c in test_dataframe.columns if c.lower() == text_column.lower()][0]
        else:
            text_column = find_column(test_dataframe, TEXT_CANDIDATES)

    label_column = None
    for candidate in LABEL_CANDIDATES:
        if candidate in train_dataframe.columns and candidate in test_dataframe.columns:
            label_column = candidate
            break
    if label_column is None:
        print("Label column not found. Attempting to derive labels for PROMISE-style data...")
        train_dataframe["__DerivedLabel"] = derive_promise_binary_labels(train_dataframe)
        test_dataframe["__DerivedLabel"]  = derive_promise_binary_labels(test_dataframe)
        label_column = "__DerivedLabel"

    multilabel = is_multilabel_series(train_dataframe[label_column])

    unique_train = pd.unique(train_dataframe[label_column].astype(str))
    if not multilabel and len(unique_train) < 2:
        print("Detected single-class labels in training set. Deriving PROMISE FR/NFR labels...")
        train_dataframe[label_column] = derive_promise_binary_labels(train_dataframe)
        test_dataframe[label_column]  = derive_promise_binary_labels(test_dataframe)
        unique_train = pd.unique(train_dataframe[label_column].astype(str))
        if len(unique_train) < 2:
            raise ValueError(
                "Training labels still have a single class after derivation. "
                "Check your PROMISE-Relabelled preprocessing or source CSV."
            )

    return train_dataframe, test_dataframe, text_column, label_column, multilabel

def train_and_evaluate(train_path: str, test_path: str, output_prefix: str):
    train_dataframe, test_dataframe, text_column, label_column, multilabel = load_data(train_path, test_path)

    print(f"Loaded training data: {train_dataframe.shape}, test data: {test_dataframe.shape}")
    print(f"Using text column: '{text_column}', label column: '{label_column}'")
    print(f"Multilabel: {multilabel}")

    X_train = train_dataframe[text_column].astype(str).fillna("")
    X_test  = test_dataframe[text_column].astype(str).fillna("")

    keep_train = X_train.str.strip() != ""
    keep_test  = X_test.str.strip() != ""
    if not keep_train.all():
        train_dataframe = train_dataframe.loc[keep_train].reset_index(drop=True)
        X_train  = X_train.loc[keep_train].reset_index(drop=True)
    if not keep_test.all():
        test_dataframe  = test_dataframe.loc[keep_test].reset_index(drop=True)
        X_test   = X_test.loc[keep_test].reset_index(drop=True)

    if multilabel:
        def parse_labels(x):
            if pd.isna(x):
                return []
            parts = [t.strip() for t in str(x).split(",")]
            return [p for p in parts if p and p.lower() not in {"none", "nan"}]

        y_train_list = train_dataframe[label_column].apply(parse_labels)
        y_test_list  = test_dataframe[label_column].apply(parse_labels)

        m_label = MultiLabelBinarizer()
        y_train = m_label.fit_transform(y_train_list)
        y_test  = m_label.transform(y_test_list)
        class_names = list(m_label.classes_)
    else:
        y_train = train_dataframe[label_column].astype(str)
        y_test  = test_dataframe[label_column].astype(str)
        class_names = sorted(y_train.unique())

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    base_logisticregression = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        class_weight=None,
        random_state=42,
    )

    if multilabel:
        model = Pipeline([
            ("tfidf", vectorizer),
            ("clf", OneVsRestClassifier(base_logisticregression))
        ])
        param_grid = {
            "clf__estimator__C": [0.5, 1.0, 2.0]
        }
    else:
        model = Pipeline([
            ("tfidf", vectorizer),
            ("clf", base_logisticregression)
        ])
        param_grid = {
            "clf__C": [0.5, 1.0, 2.0]
        }

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
        error_score="raise"
    )

    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    best_model = grid.best_estimator_

    y_prediction = best_model.predict(X_test)

    print("\n Evaluation Report")
    if multilabel:
        print(classification_report(y_test, y_prediction, target_names=class_names, zero_division=0))
        macro_f1 = f1_score(y_test, y_prediction, average="macro", zero_division=0)
        print(f"Macro F1: {macro_f1:.4f}")
    else:
        print(classification_report(y_test, y_prediction, zero_division=0))
        accuracy = accuracy_score(y_test, y_prediction)
        macro_f1 = f1_score(y_test, y_prediction, average="macro", zero_division=0)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

    os.makedirs("results/models", exist_ok=True)
    import joblib
    model_path = f"results/models/logreg_{output_prefix}.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to {model_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(here, "..", "..", "data")

    datasets = {
        "PROMISE": {
            "train": os.path.join(data_directory, "PROMISE-Relabelled", "processed", "train.csv"),
            "test":  os.path.join(data_directory, "PROMISE-Relabelled", "processed", "test.csv"),
            "prefix": "promise"
        },
        "PURE": {
            "train": os.path.join(data_directory, "PURE", "processed", "train.csv"),
            "test":  os.path.join(data_directory, "PURE", "processed", "test.csv"),
            "prefix": "pure"
        }
    }

    for name, cfg in datasets.items():
        print(f"\n Training Logistic Regression on {name} dataset")
        try:
            train_and_evaluate(cfg["train"], cfg["test"], cfg["prefix"])
        except Exception as e:
            print(f"Error processing {name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
