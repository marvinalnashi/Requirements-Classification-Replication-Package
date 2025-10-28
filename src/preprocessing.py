"""
Preprocessing for PURE and PROMISE-Relabelled datasets.
Performs text cleaning, normalisation, and saves processed CSVs for model training.
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()                                        # lowercase
    text = re.sub(r"http\S+", " ", text)           # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()       # normalise whitespace
    return text

def preprocess_dataset(path: str, text_column: str, label_column: str, output_directory: str,
                       test_size: float = 0.2, random_state: int = 42, multilabel_columns=None):
    try:
        dataframe = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    except UnicodeDecodeError:
        dataframe = pd.read_csv(path, sep=";", encoding="latin1")
    except Exception:
        dataframe = pd.read_csv(path, engine="python", sep=None, encoding="latin1")

    if list(dataframe.columns)[0].strip().upper() in ["A", "B", "C"]:
        try:
            dataframe = pd.read_csv(path, sep=";", header=1, encoding="utf-8-sig")
        except UnicodeDecodeError:
            dataframe = pd.read_csv(path, sep=";", header=1, encoding="latin1")

    print(f"Loaded columns from {os.path.basename(path)}: {list(dataframe.columns)}")

    if multilabel_columns:
        missing = [c for c in multilabel_columns if c not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing expected columns {missing} in {path}")

        dataframe["Labels"] = dataframe[multilabel_columns].apply(
            lambda row: ",".join([col for col in multilabel_columns if row[col] == 1]) or "None",
            axis=1
        )
        label_column = "Labels"

    if text_column not in dataframe.columns or label_column not in dataframe.columns:
        raise ValueError(f"Columns {text_column} or {label_column} not found in {path}")

    dataframe = dataframe.dropna(subset=[text_column, label_column])
    dataframe[text_column] = dataframe[text_column].apply(clean_text)

    if not multilabel_columns:
        dataframe[label_column] = dataframe[label_column].apply(
            lambda x: "NFR" if "non" in str(x).lower() else "FR"
        )

    try:
        train_dataframe, test_dataframe = train_test_split(
            dataframe, test_size=test_size, stratify=dataframe[label_column], random_state=random_state
        )
    except ValueError:
        train_dataframe, test_dataframe = train_test_split(
            dataframe, test_size=test_size, random_state=random_state
        )

    os.makedirs(output_directory, exist_ok=True)
    train_path = os.path.join(output_directory, "train.csv")
    test_path = os.path.join(output_directory, "test.csv")
    train_dataframe.to_csv(train_path, index=False)
    test_dataframe.to_csv(test_path, index=False)

    print(f"Preprocessed {os.path.basename(path)}")
    print(f" -> Train: {train_dataframe.shape}, Test: {test_dataframe.shape}")
    print(f" Saved to {output_directory}/\n")

    return train_path, test_path

def main():
    base_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(base_directory, "..", "data")

    datasets = {
        "PROMISE": {
            "path": os.path.join(data_directory, "PROMISE-Relabelled", "promise-reclass.csv"),
            "text_column": "RequirementText",
            "label_column": "Class",
            "output_directory": os.path.join(data_directory, "PROMISE-Relabelled", "processed")
        },
        "PURE": {
            "path": os.path.join(data_directory, "PURE", "Pure_Annotate_Dataset.csv"),
            "text_column": "sentence",
            "label_column": None,
            "output_directory": os.path.join(data_directory, "PURE", "processed"),
            "multilabel_columns": ["security", "reliability", "NFR_boolean"]
        }
    }

    for name, cfg in datasets.items():
        print(f"\n Processing {name} dataset")
        try:
            if "multilabel_columns" in cfg:
                preprocess_dataset(**cfg)
            else:
                preprocess_dataset(**{key: value for key, value in cfg.items() if key != "multilabel_columns"})
        except Exception as e:
            print(f"Error processing {name}: {e}")


if __name__ == "__main__":
    main()
