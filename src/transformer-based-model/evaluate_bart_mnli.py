"""
Zero-shot requirement classification using facebook/bart-large-mnli.
This method can be slow if using a cpu. It is recommended to utilise a gpu using CUDA instead if supported.
- PROMISE-Relabelled: single-label
- PURE: multi-label

NO fine-tuning required.
"""

import os
import sys
from typing import List, Dict

import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_NAME = "facebook/bart-large-mnli"
HYPOTHESIS = "This requirement is about {}."
MAX_LEN = 256
BATCH_SIZE = 64

PROMISE_LABELS = {
    "functional requirements": "FR",
    "non-functional requirements": "NFR",
}
PURE_LABELS = {
    "security": "security",
    "reliability": "reliability",
    "non-functional requirements": "NFR_boolean",
}
PURE_THRESHOLD = 0.50

TEXT_CANDIDATES = [
    "RequirementText", "requirementtext", "sentence", "text", "requirement", "req_text", "req"
]

PROMISE_INDICATORS = ["IsFunctional", "IsQuality", "IsOnlyFunctional", "IsOnlyQuality", "F+Q", "NotR"]


def get_device():
    want_cuda = os.environ.get("USE_CUDA", "").strip().lower() in {"1", "true", "yes"}
    has_cuda = torch.cuda.is_available()
    if want_cuda and not has_cuda:
        print("USE_CUDA requested but PyTorch has no CUDA, falling back to CPU.")
    return 0 if (want_cuda and has_cuda) else -1


def load_split_paths() -> Dict[str, Dict[str, str]]:
    here = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(here, "..", "..", "data")
    return {
        "PROMISE": {
            "test": os.path.join(data_directory, "PROMISE-Relabelled", "processed", "test.csv"),
        },
        "PURE": {
            "test": os.path.join(data_directory, "PURE", "processed", "test.csv"),
        }
    }


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

    class_column = None
    for candidate in ["Class", "class"]:
        if candidate in dataframe.columns:
            class_column = candidate
            break
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


def parse_multilabel_column(s: pd.Series) -> List[List[str]]:
    def parse_one(x):
        if pd.isna(x):
            return []
        parts = [t.strip() for t in str(x).split(",")]
        return [p for p in parts if p and p.lower() not in {"none", "nan"}]
    return s.apply(parse_one).tolist()


def build_pipeline(device: int):
    token = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if device == -1:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, dtype=torch.float16
        ).to("cuda:0")

    zeroshot = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=token,
        device=0 if device != -1 else -1
    )
    return zeroshot


def zero_shot_single_batched(texts: List[str], candidate_map: Dict[str, str], zeroshot, batch_size=BATCH_SIZE) -> List[str]:
    cand = list(candidate_map.keys())
    predictions: List[str] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="PROMISE batches"):
        chunk = texts[i:i+batch_size]
        output = zeroshot(
            sequences=chunk,
            candidate_labels=cand,
            hypothesis_template=HYPOTHESIS,
            multi_label=False,
            truncation=True,
            max_length=MAX_LEN
        )
        if isinstance(output, dict):
            output = [output]
        for o in output:
            predictions.append(candidate_map[o["labels"][0]])
    return predictions


def zero_shot_multi_batched(texts: List[str], candidate_map: Dict[str, str], zeroshot,
                            threshold=PURE_THRESHOLD, batch_size=BATCH_SIZE) -> List[List[str]]:
    candidate = list(candidate_map.keys())
    all_predictions: List[List[str]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="PURE batches"):
        chunk = texts[i:i+batch_size]
        output = zeroshot(
            sequences=chunk,
            candidate_labels=candidate,
            hypothesis_template=HYPOTHESIS,
            multi_label=True,
            truncation=True,
            max_length=MAX_LEN
        )
        if isinstance(output, dict):
            output = [output]
        for o in output:
            chosen = [candidate_map[label] for label, score in zip(o["labels"], o["scores"]) if score >= threshold]
            all_predictions.append(chosen)
    return all_predictions


def evaluate_promise(test_csv: str, zeroshot) -> None:
    dataframe = pd.read_csv(test_csv)
    text_column = find_text_column(dataframe)

    y_true = derive_promise_labels(dataframe).astype(str).tolist()
    X = dataframe[text_column].astype(str).fillna("").tolist()

    y_prediction = zero_shot_single_batched(X, PROMISE_LABELS, zeroshot)

    print("\n PROMISE (BART-MNLI)")
    print(classification_report(y_true, y_prediction, zero_division=0))
    accuracy = accuracy_score(y_true, y_prediction)
    macro_f1 = f1_score(y_true, y_prediction, average="macro", zero_division=0)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_prediction": y_prediction}).to_csv(
        "results/tables/bartmnli_promise_predictions.csv", index=False
    )


def evaluate_pure(test_csv: str, zeroshot, threshold=PURE_THRESHOLD) -> None:
    dataframe = pd.read_csv(test_csv)
    text_column = find_text_column(dataframe)

    label_column = None
    for c in ["Labels", "labels"]:
        if c in dataframe.columns:
            label_column = c
            break
    if label_column is None:
        raise ValueError("PURE test.csv missing 'Labels' column (expected after preprocessing).")

    y_true_lists = parse_multilabel_column(dataframe[label_column])
    X = dataframe[text_column].astype(str).fillna("").tolist()

    y_prediction_lists = zero_shot_multi_batched(X, PURE_LABELS, zeroshot, threshold=threshold)

    m_label = MultiLabelBinarizer()
    m_label.fit([list(PURE_LABELS.values())])
    y_true = m_label.transform(y_true_lists)
    y_prediction = m_label.transform(y_prediction_lists)
    class_names = list(m_label.classes_)

    print("\n PURE (BART-MNLI)")
    print(classification_report(y_true, y_prediction, target_names=class_names, zero_division=0))
    macro_f1 = f1_score(y_true, y_prediction, average="macro", zero_division=0)
    print(f"Macro F1: {macro_f1:.4f}")

    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame({
        "y_true": [",".join(lbls) for lbls in y_true_lists],
        "y_prediction": [",".join(lbls) for lbls in y_prediction_lists],
    }).to_csv("results/tables/bartmnli_pure_predictions.csv", index=False)


def main():
    paths = load_split_paths()
    device = get_device()
    print(f"Using device: {'cuda:0' if device == 0 else 'cpu'}")

    zeroshot = build_pipeline(device)

    try:
        print("\n Evaluating PROMISE-Relabelled (FR vs NFR)")
        evaluate_promise(paths["PROMISE"]["test"], zeroshot)
    except Exception as e:
        print(f"PROMISE evaluation error: {e}", file=sys.stderr)

    try:
        print("\n Evaluating PURE (security, reliability, NFR_boolean)")
        evaluate_pure(paths["PURE"]["test"], zeroshot, threshold=PURE_THRESHOLD)
    except Exception as e:
        print(f"PURE evaluation error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
