"""
Automatically detects model metrics JSONs in results/stats/ and produces:
- Per-class F1 bar charts for PROMISE and PURE
- Summary charts (macro-F1 + accuracy/micro-F1) for PROMISE and PURE
- Overall Model Comparison (RQ1)
- Dataset Impact (RQ2)
- Cross-Dataset Generalisation (RQ3) with graceful fallback to a proxy
- A CSV table with the key numbers

Flags:
  --stats_directory  results/stats
  --fig_directory    results/figures
  --tables_directory results/tables

Usage:
  python src/evaluation/generate_all_figures.py
  python src/evaluation/generate_all_figures.py --stats_directory ../../results/stats
"""

import argparse
import glob
import json
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def safe_series_max(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    return float(s.max()) if not s.empty else 0.0


def ensure_directories(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_name_interpreter(raw: str) -> str:
    name = raw.strip().lower()
    mapping = {
        "svm": "SVM",
        "logreg": "LogReg",
        "bartmnli": "BART",
        "sbert": "SBERT",
    }
    return mapping.get(name, raw.upper())


def extract_perclass_f1(report_dictionary):
    output = {}
    for key, value in report_dictionary.items():
        if isinstance(value, dict) and key not in {"macro avg", "weighted avg", "accuracy"}:
            output[key] = float(value.get("f1-score", 0.0))
    return output


def discover_metrics(stats_directory):
    promise, pure = {}, {}
    for path in glob.glob(os.path.join(stats_directory, "*.json")):
        base = os.path.basename(path)
        if base.endswith("_promise_metrics.json"):
            model = base.replace("_promise_metrics.json", "")
            promise[model] = path
        elif base.endswith("_pure_metrics.json"):
            model = base.replace("_pure_metrics.json", "")
            pure[model] = path
    return promise, pure


def discover_cross_metrics(stats_directory):
    cross_map = defaultdict(dict)
    for path in glob.glob(os.path.join(stats_directory, "*.json")):
        base = os.path.basename(path).lower()
        if "trainpromise_testpure" in base:
            model = base.split("_trainpromise_testpure_metrics.json")[0]
            cross_map[model][("PROMISE", "PURE")] = path
        elif "trainpure_testpromise" in base:
            model = base.split("_trainpure_testpromise_metrics.json")[0]
            cross_map[model][("PURE", "PROMISE")] = path
    return cross_map


def plot_perclass_f1(models_to_json, dataset_name, figure_directory):
    if not models_to_json:
        print(f"[WARN] No JSONs found for {dataset_name}, skipping per-class F1 plot.")
        return None

    perclass = {}
    all_labels = set()
    for model, path in models_to_json.items():
        data = load_json(path)
        report = data.get("report", {})
        f1s = extract_perclass_f1(report)
        perclass[model_name_interpreter(model)] = f1s
        all_labels |= set(f1s.keys())

    if not all_labels:
        print(f"[WARN] No per-class labels found for {dataset_name}.")
        return None

    labels = sorted(all_labels)
    x = np.arange(len(labels))
    width = 0.12 if len(perclass) > 6 else 0.15

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10, 5))

    model_names = sorted(perclass.keys())
    for i, name in enumerate(model_names):
        values = [perclass[name].get(label, 0.0) for label in labels]
        axis.bar(x + i * width, values, width, label=name)

    axis.set_ylabel("F1-score")
    axis.set_ylim(0, 1.0)
    axis.set_title(f"{dataset_name}: Per-class F1 Comparison")
    axis.set_xticks(x + width * (len(model_names) - 1) / 2)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.legend(ncol=2, fontsize=9)
    figure.tight_layout()

    output = os.path.join(figure_directory, f"{dataset_name.lower()}_perclass_f1.png")
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(f"[OK] Saved {output}")
    return output


def plot_summary(models_to_json, dataset_name, figure_directory, is_pure=False):
    if not models_to_json:
        print(f"[WARN] No JSONs found for {dataset_name}, skipping summary plot.")
        return None

    plt.style.use("seaborn-v0_8-whitegrid")
    names, macro, other = [], [], []
    for model, path in sorted(models_to_json.items()):
        d = load_json(path)
        names.append(model_name_interpreter(model))
        macro.append(float(d.get("macro_f1", d.get("macro F1", 0.0))))
        if is_pure:
            other.append(float(d.get("micro_f1", 0.0)))
        else:
            other.append(float(d.get("accuracy", 0.0)))

    x = np.arange(len(names))
    width = 0.35
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width/2, macro, width, label="Macro-F1")
    axis.bar(x + width/2, other, width, label=("Micro-F1" if is_pure else "Accuracy"))
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title(f"{dataset_name}: Model Performance Summary")
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=20, ha="right")
    axis.legend()
    figure.tight_layout()

    output = os.path.join(figure_directory, f"{dataset_name.lower()}_summary.png")
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(f"[OK] Saved {output}")
    return output


def write_summary_table(promise_jsons, pure_jsons, tables_directory):
    rows = []
    for model, path in sorted(promise_jsons.items()):
        d = load_json(path)
        rows.append(OrderedDict([
            ("dataset", "PROMISE"),
            ("model", model_name_interpreter(model)),
            ("macro_f1", round(float(d.get("macro_f1", 0.0)), 4)),
            ("accuracy", round(float(d.get("accuracy", 0.0)), 4)),
            ("micro_f1", ""),
        ]))
    for model, path in sorted(pure_jsons.items()):
        d = load_json(path)
        rows.append(OrderedDict([
            ("dataset", "PURE"),
            ("model", model_name_interpreter(model)),
            ("macro_f1", round(float(d.get("macro_f1", 0.0)), 4)),
            ("accuracy", ""),
            ("micro_f1", round(float(d.get("micro_f1", 0.0)), 4)),
        ]))
    if not rows:
        print("[WARN] No JSONs found to write summary table.")
        return None

    dataframe = pd.DataFrame(rows)
    output = os.path.join(tables_directory, "metrics_summary.csv")
    dataframe.to_csv(output, index=False)
    print(f"[OK] Wrote {output}")
    return output


def collect_macro_f1(promise_jsons, pure_jsons):
    records = []
    for model, path in promise_jsons.items():
        d = load_json(path)
        records.append({"Model": model_name_interpreter(model), "Dataset": "PROMISE", "MacroF1": float(d.get("macro_f1", 0.0))})
    for model, path in pure_jsons.items():
        d = load_json(path)
        records.append({"Model": model_name_interpreter(model), "Dataset": "PURE", "MacroF1": float(d.get("macro_f1", 0.0))})
    return pd.DataFrame.from_records(records)


def plot_overall_model_comparison(promise_jsons, pure_jsons, figure_directory):
    """
    RQ1: Overall comparison of model types (models on X-axis, bars for datasets).
    """
    dataframe = collect_macro_f1(promise_jsons, pure_jsons)
    if dataframe.empty:
        print("[WARN] No metrics for overall model comparison.")
        return None

    preferred = ["SVM", "LogReg", "BART", "SBERT"]
    model_order = sorted(
        dataframe["Model"].unique(),
        key=lambda m: preferred.index(m) if m in preferred else 999
    )

    datasets = ["PROMISE", "PURE"]
    x = np.arange(len(model_order))
    width = 0.35

    series = {}
    for dataset in datasets:
        series[dataset] = [
            safe_series_max(dataframe[(dataframe["Model"] == m) & (dataframe["Dataset"] == dataset)]["MacroF1"])
            for m in model_order
        ]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, series["PROMISE"], width, label="PROMISE")
    axis.bar(x + width / 2, series["PURE"], width, label="PURE")

    axis.set_title("Overall Model Comparison (Macro-F1)")
    axis.set_ylabel("Macro-F1")
    axis.set_ylim(0, 1.0)
    axis.set_xticks(x)
    axis.set_xticklabels(model_order, rotation=15, ha="right")
    axis.legend()
    figure.tight_layout()

    output = os.path.join(figure_directory, "overall_model_comparison.png")
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(f"[OK] Saved {output}")
    return output


def plot_dataset_impact(promise_jsons, pure_jsons, figure_directory):
    """
    RQ2: Dataset impact — show how each model's Macro-F1 changes between PROMISE and PURE.
    Render as a slope chart (two points per model connected by a line).
    """
    dataframe = collect_macro_f1(promise_jsons, pure_jsons)
    if dataframe.empty:
        print("[WARN] No metrics for dataset impact.")
        return None

    preferred = ["SVM", "LogReg", "BART", "SBERT"]
    model_order = sorted(
        dataframe["Model"].unique(),
        key=lambda m: preferred.index(m) if m in preferred else 999
    )

    y_prom = [
        safe_series_max(dataframe[(dataframe["Model"] == m) & (dataframe["Dataset"] == "PROMISE")]["MacroF1"])
        for m in model_order
    ]
    y_pure = [
        safe_series_max(dataframe[(dataframe["Model"] == m) & (dataframe["Dataset"] == "PURE")]["MacroF1"])
        for m in model_order
    ]

    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10, 5))

    x0, x1 = 0, 1
    for i, m in enumerate(model_order):
        axis.plot([x0, x1], [y_prom[i], y_pure[i]], marker="o")
        midx = (x0 + x1) / 2.0
        midy = (y_prom[i] + y_pure[i]) / 2.0
        axis.text(midx, midy, m, fontsize=9, ha="center", va="bottom")

    axis.set_xticks([x0, x1])
    axis.set_xticklabels(["PROMISE", "PURE"])
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Macro-F1")
    axis.set_title("Dataset Impact on Macro-F1 (PROMISE → PURE)")
    figure.tight_layout()

    output = os.path.join(figure_directory, "dataset_impact.png")
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(f"[OK] Saved {output}")
    return output


from collections import defaultdict
import numpy as np
import os

def plot_cross_dataset_generalisation(promise_jsons, pure_jsons, cross_map, figure_directory):
    """
    RQ3: Cross-dataset generalisation.

    TRUE mode:
      Uses *_trainPROMISE_testPURE_metrics.json and *_trainPURE_testPROMISE_metrics.json
      and plots Δ = Within(PROMISE)−Cross(PROMISE→PURE) and Within(PURE)−Cross(PURE→PROMISE).

    PROXY mode:
      If TRUE cross files are missing, plot a single Δ per model: |MacroF1(PROMISE) − MacroF1(PURE)|.
    """
    within = {}
    for model_raw, path in promise_jsons.items():
        model = model_name_interpreter(model_raw)  # or nice_model_name(...)
        within.setdefault(model, {})["PROMISE"] = float(load_json(path).get("macro_f1", 0.0))
    for model_raw, path in pure_jsons.items():
        model = model_name_interpreter(model_raw)  # or nice_model_name(...)
        within.setdefault(model, {})["PURE"] = float(load_json(path).get("macro_f1", 0.0))

    has_true_cross = False
    cross_scores = defaultdict(dict)
    for model_raw, transfers in cross_map.items():
        model = model_name_interpreter(model_raw)  # or nice_model_name(...)
        if ("PROMISE", "PURE") in transfers:
            cross_scores[model]["PROMISE->PURE"] = float(load_json(transfers[("PROMISE", "PURE")]).get("macro_f1", 0.0))
            has_true_cross = True
        if ("PURE", "PROMISE") in transfers:
            cross_scores[model]["PURE->PROMISE"] = float(load_json(transfers[("PURE", "PROMISE")]).get("macro_f1", 0.0))
            has_true_cross = True

    preferred = ["SVM", "LogReg", "BART", "SBERT"]
    models_sorted = sorted(within.keys(), key=lambda m: preferred.index(m) if m in preferred else 999)

    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    x = np.arange(len(models_sorted))
    width = 0.35
    figure, axis = plt.subplots(figsize=(10, 5))

    if has_true_cross:
        bars_left, bars_right = [], []
        for m in models_sorted:
            within_promise = within.get(m, {}).get("PROMISE", np.nan)
            within_pure = within.get(m, {}).get("PURE", np.nan)
            cross_promise_pure = cross_scores.get(m, {}).get("PROMISE->PURE", np.nan)
            cross_pure_promise = cross_scores.get(m, {}).get("PURE->PROMISE", np.nan)

            d1 = (within_promise - cross_promise_pure) if (not np.isnan(within_promise) and not np.isnan(cross_promise_pure)) else np.nan
            d2 = (within_pure - cross_pure_promise) if (not np.isnan(within_pure) and not np.isnan(cross_pure_promise)) else np.nan
            bars_left.append(d1 if not np.isnan(d1) else 0.0)
            bars_right.append(d2 if not np.isnan(d2) else 0.0)

        axis.bar(x - width/2, bars_left, width, label="PROMISE→PURE")
        axis.bar(x + width/2, bars_right, width, label="PURE→PROMISE")
        axis.set_ylabel("ΔMacro-F1 (Within − Transfer)")
        title = "Cross-Dataset Generalisation ΔMacro-F1"
        upper = max(bars_left + bars_right) if (bars_left or bars_right) else 0.0
    else:
        deltas = []
        for m in models_sorted:
            within_promise = within.get(m, {}).get("PROMISE", np.nan)
            within_pure = within.get(m, {}).get("PURE", np.nan)
            d = abs(within_promise - within_pure) if (not np.isnan(within_promise) and not np.isnan(within_pure)) else 0.0
            deltas.append(d)

        axis.bar(x, deltas, width, label="|PROMISE − PURE|")
        axis.set_ylabel("ΔMacro-F1 (|PROMISE − PURE|)")
        title = "Cross-Dataset Generalisation ΔMacro-F1 (proxy)"
        upper = max(deltas) if deltas else 0.0

    axis.set_title(title)
    axis.set_ylim(0, max(0.01, min(1.0, upper + 0.05)))
    axis.set_xticks(x)
    axis.set_xticklabels(models_sorted, rotation=15, ha="right")
    axis.legend()
    figure.tight_layout()

    figure_name = "cross_dataset_generalisation.png" if has_true_cross else "cross_dataset_generalisation_proxy.png"
    output = os.path.join(figure_directory, figure_name)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(f"[OK] Saved {output} {'(TRUE cross)' if has_true_cross else '(PROXY)'}")
    return output


def main():
    argumentparser = argparse.ArgumentParser()
    argumentparser.add_argument("--stats_directory", default="../../results/stats")
    argumentparser.add_argument("--fig_directory", default="../../results/figures")
    argumentparser.add_argument("--tables_directory", default="../../results/tables")
    arguments = argumentparser.parse_args()

    ensure_directories(arguments.stats_directory, arguments.fig_directory, arguments.tables_directory)

    promise_jsons, pure_jsons = discover_metrics(arguments.stats_directory)
    if not promise_jsons and not pure_jsons:
        print(f"[WARN] No metrics JSONs found in {arguments.stats_directory}. Run your evaluators first.")
        return

    plot_perclass_f1(promise_jsons, "PROMISE", arguments.fig_directory)
    plot_perclass_f1(pure_jsons, "PURE", arguments.fig_directory)

    plot_summary(promise_jsons, "PROMISE", arguments.fig_directory, is_pure=False)
    plot_summary(pure_jsons, "PURE", arguments.fig_directory, is_pure=True)

    write_summary_table(promise_jsons, pure_jsons, arguments.tables_directory)

    plot_overall_model_comparison(promise_jsons, pure_jsons, arguments.fig_directory)                 # RQ1
    plot_dataset_impact(promise_jsons, pure_jsons, arguments.fig_directory)                           # RQ2

    cross_map = discover_cross_metrics(arguments.stats_directory)
    plot_cross_dataset_generalisation(promise_jsons, pure_jsons, cross_map, arguments.fig_directory)  # RQ3

    print("[DONE] All figures generated.")


if __name__ == "__main__":
    main()
