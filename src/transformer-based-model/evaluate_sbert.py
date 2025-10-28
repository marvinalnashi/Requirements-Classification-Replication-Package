"""
Zero-training requirement classification using SBERT embeddings.
This method is fast. CUDA is recommended but not required for this method.
- PROMISE-Relabelled: single-label
- PURE: multi-label

No fine-tuning required.
"""

import os
import sys
from typing import List, Dict

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

SBERT_MODEL = os.environ.get("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_LEN = int(os.environ.get("SBERT_MAX_LEN", "256"))
BATCH_SIZE = int(os.environ.get("SBERT_BATCH", "256"))
PURE_THRESHOLD = float(os.environ.get("SBERT_THRESHOLD", "0.35"))

PROMISE_LABELS = {
    "functional requirement": "FR",
    "non-functional requirement": "NFR",
}
PURE_LABELS = {
    "security requirement": "security",
    "reliability requirement": "reliability",
    "non-functional requirement": "NFR_boolean",
}

TEXT_CANDIDATES = [
    "RequirementText", "requirementtext", "sentence", "text", "requirement", "req_text", "req"
]

PROMISE_INDICATORS = ["IsFunctional", "IsQuality", "IsOnlyFunctional", "IsOnlyQuality", "F+Q", "NotR"]


def get_device() -> str:
    want_cuda = os.environ.get("USE_CUDA", "").strip().lower() in {"1", "true", "yes"}
    has_cuda = torch.cuda.is_available()
    if want_cuda and not has_cuda:
        print("USE_CUDA requested but PyTorch has no CUDA, falling back to CPU.")
    return "cuda" if (want_cuda and has_cuda) else "cpu"


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
    for c in ["Class", "class"]:
        if c in dataframe.columns:
            class_column = c
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


def ensure_results_directories():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)


def build_sbert(device_str: str) -> SentenceTransformer:
    model = SentenceTransformer(SBERT_MODEL, device=device_str)
    try:
        model.max_seq_length = MAX_LEN
    except Exception:
        pass
    return model


def encode_texts(model: SentenceTransformer, texts: List[str]) -> torch.Tensor:
    embeds = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return embeds


def evaluate_promise(test_csv: str, model: SentenceTransformer) -> None:
    dataframe = pd.read_csv(test_csv)
    text_column = find_text_column(dataframe)
    X = dataframe[text_column].astype(str).fillna("").tolist()
    y_true = derive_promise_labels(dataframe).astype(str).tolist()

    label_phrases = list(PROMISE_LABELS.keys())
    label_keys = [PROMISE_LABELS[p] for p in label_phrases]
    label_embeds = encode_texts(model, label_phrases)

    text_embeds = encode_texts(model, X)

    sims = (text_embeds @ label_embeds.T)
    top_idx = sims.argmax(dim=1).cpu().tolist()
    y_prediction = [label_keys[i] for i in top_idx]

    print("\n PROMISE (SBERT)")
    print(classification_report(y_true, y_prediction, zero_division=0))
    accuracy = accuracy_score(y_true, y_prediction)
    macro_f1 = f1_score(y_true, y_prediction, average="macro", zero_division=0)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    ensure_results_directories()
    pd.DataFrame({"y_true": y_true, "y_prediction": y_prediction}).to_csv(
        "results/tables/sbert_promise_predictions.csv", index=False
    )


def evaluate_pure(test_csv: str, model: SentenceTransformer, threshold: float = PURE_THRESHOLD) -> None:
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

    label_phrases = list(PURE_LABELS.keys())
    label_keys = [PURE_LABELS[p] for p in label_phrases]
    label_embeds = encode_texts(model, label_phrases)

    text_embeds = encode_texts(model, X)

    sims = (text_embeds @ label_embeds.T)

    y_prediction_lists: List[List[str]] = []
    for row in sims:
        chosen = [label_keys[j] for j, score in enumerate(row.tolist()) if score >= threshold]
        y_prediction_lists.append(chosen)

    m_label = MultiLabelBinarizer()
    m_label.fit([label_keys])
    y_true = m_label.transform(y_true_lists)
    y_prediction = m_label.transform(y_prediction_lists)
    class_names = list(m_label.classes_)

    print("\n PURE (SBERT)")
    print(classification_report(y_true, y_prediction, target_names=class_names, zero_division=0))
    macro_f1 = f1_score(y_true, y_prediction, average="macro", zero_division=0)
    print(f"Macro F1: {macro_f1:.4f}")

    ensure_results_directories()
    pd.DataFrame({
        "y_true": [",".join(lbls) for lbls in y_true_lists],
        "y_prediction": [",".join(lbls) for lbls in y_prediction_lists],
    }).to_csv("results/tables/sbert_pure_predictions.csv", index=False)


def main():
    paths = load_split_paths()
    device_string = get_device()
    print(f"SBERT device: {device_string}")

    model = build_sbert(device_string)

    try:
        print("\n Evaluating PROMISE-Relabelled (FR vs NFR)")
        evaluate_promise(paths["PROMISE"]["test"], model)
    except Exception as e:
        print(f"PROMISE evaluation error: {e}", file=sys.stderr)

    try:
        print("\n Evaluating PURE (security, reliability, NFR_boolean)")
        evaluate_pure(paths["PURE"]["test"], model, threshold=PURE_THRESHOLD)
    except Exception as e:
        print(f"PURE evaluation error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
