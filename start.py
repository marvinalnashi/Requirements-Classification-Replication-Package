"""
GUI launcher for the Requirements Classification replication package.

Main menu options:
  1) Run full pipeline (progress window with live output + progress bar)
  2) Clean generated files (runs src/clean_generated.py)
  3) Open README.md
  4) Exit
"""

import os
import sys
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

REPOSITORY_ROOT = Path(__file__).resolve().parent

PREPROCESS = REPOSITORY_ROOT / "src" / "preprocessing.py"
TRAIN_SVM = REPOSITORY_ROOT / "src" / "shallow-model" / "train_svm.py"
TRAIN_LOGISTICREGRESSION = REPOSITORY_ROOT / "src" / "shallow-model" / "train_logreg.py"
PREDICT_SHALLOW = REPOSITORY_ROOT / "src" / "shallow-model" / "predict_shallow.py"
EVALUATE_BART = REPOSITORY_ROOT / "src" / "transformer-based-model" / "evaluate_bart_mnli.py"
EVALUATE_SBERT = REPOSITORY_ROOT / "src" / "transformer-based-model" / "evaluate_sbert.py"
EVALUATE_PREDICTION = REPOSITORY_ROOT / "src" / "evaluation" / "evaluate_predictions.py"
GENERATE_FIGURES = REPOSITORY_ROOT / "src" / "generate_all_figures.py"
CLEAN = REPOSITORY_ROOT / "src" / "clean_generated.py"
README = REPOSITORY_ROOT / "README.md"

RESULT_DIRECTORY = [
    REPOSITORY_ROOT / "results"
]

def pipeline_commands():
    py = sys.executable
    steps = []

    steps.append([py, str(PREPROCESS)])

    steps.append([py, str(TRAIN_SVM)])
    steps.append([py, str(TRAIN_LOGISTICREGRESSION)])

    steps.append([py, str(PREDICT_SHALLOW), "--model", str(REPOSITORY_ROOT / "results" / "models" / "svm_promise.pkl"), "--dataset", "PROMISE"])
    steps.append([py, str(PREDICT_SHALLOW), "--model", str(REPOSITORY_ROOT / "results" / "models" / "svm_pure.pkl"), "--dataset", "PURE"])
    steps.append([py, str(PREDICT_SHALLOW), "--model", str(REPOSITORY_ROOT / "results" / "models" / "logreg_promise.pkl"), "--dataset", "PROMISE"])
    steps.append([py, str(PREDICT_SHALLOW), "--model", str(REPOSITORY_ROOT / "results" / "models" / "logreg_pure.pkl"), "--dataset", "PURE"])

    steps.append([py, str(EVALUATE_BART)])
    steps.append([py, str(EVALUATE_SBERT)])

    steps.append([py, str(EVALUATE_PREDICTION), "--prediction", str(REPOSITORY_ROOT / "results" / "tables" / "svm_promise_promise_predictions.csv"), "--dataset", "PROMISE", "--name", "svm"])
    steps.append([py, str(EVALUATE_PREDICTION), "--prediction", str(REPOSITORY_ROOT / "results" / "tables" / "svm_pure_pure_predictions.csv"), "--dataset", "PURE", "--name", "svm"])
    steps.append([py, str(EVALUATE_PREDICTION), "--prediction", str(REPOSITORY_ROOT / "results" / "tables" / "logreg_promise_promise_predictions.csv"), "--dataset", "PROMISE", "--name", "logreg"])
    steps.append([py, str(EVALUATE_PREDICTION), "--prediction", str(REPOSITORY_ROOT / "results" / "tables" / "logreg_pure_pure_predictions.csv"), "--dataset", "PURE", "--name", "logreg"])
    steps.append([py, str(EVALUATE_PREDICTION), "--prediction", str(REPOSITORY_ROOT / "results" / "tables" / "bartmnli_promise_predictions.csv"), "--dataset", "PROMISE", "--name", "bartmnli"])
    steps.append([py, str(EVALUATE_PREDICTION), "--prediction", str(REPOSITORY_ROOT / "results" / "tables" / "bartmnli_pure_predictions.csv"), "--dataset", "PURE", "--name", "bartmnli"])
    steps.append([py, str(EVALUATE_PREDICTION), "--prediction", str(REPOSITORY_ROOT / "results" / "tables" / "sbert_promise_predictions.csv"), "--dataset", "PROMISE", "--name", "sbert"])
    steps.append([py, str(EVALUATE_PREDICTION), "--prediction", str(REPOSITORY_ROOT / "results" / "tables" / "sbert_pure_predictions.csv"), "--dataset", "PURE", "--name", "sbert"])

    steps.append([py, str(GENERATE_FIGURES)])
    return steps

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Requirements Classification Replication Package")
        self.geometry("600x400")
        self.resizable(False, False)

        ttk.Style().theme_use("clam")

        title = ttk.Label(self, text="Replication Package", font=("Segoe UI", 16, "bold"))
        title.pack(pady=(20, 10))

        self.use_cuda_var = tk.BooleanVar(value=False)
        cuda_frame = ttk.Frame(self)
        cuda_frame.pack(pady=(0, 8))
        ttk.Checkbutton(cuda_frame, text="Use CUDA (if available)", variable=self.use_cuda_var).pack()

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Run full pipeline", width=28, command=self.open_runner).grid(row=0, column=0, padx=8, pady=6)
        ttk.Button(btn_frame, text="Clean generated files", width=28, command=self.clean_generated).grid(row=1, column=0, padx=8, pady=6)
        ttk.Button(btn_frame, text="Open README.md", width=28, command=self.open_readme).grid(row=2, column=0, padx=8, pady=6)
        ttk.Button(btn_frame, text="Exit", width=28, command=self.destroy).grid(row=3, column=0, padx=8, pady=6)

        self.status = ttk.Label(self, text="Ready.", anchor="w")
        self.status.pack(side="bottom", fill="x", padx=8, pady=6)

    def open_runner(self):
        RunnerWindow(self, use_cuda=self.use_cuda_var.get())

    def clean_generated(self):
        if not CLEAN.exists():
            messagebox.showerror("Error", f"clean_generated.py not found:\n{CLEAN}")
            return
        self.status.config(text="Cleaning generated files…")
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run([sys.executable, str(CLEAN)], cwd=str(REPOSITORY_ROOT),
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
            if result.returncode == 0:
                messagebox.showinfo("Clean", "Generated files removed and folders reset.")
            else:
                messagebox.showerror("Clean error", (result.stdout or "") + "\n" + (result.stderr or ""))
        except Exception as e:
            messagebox.showerror("Clean error", str(e))
        finally:
            self.status.config(text="Ready.")

    def open_readme(self):
        try:
            if README.exists():
                if os.name == "nt":
                    os.startfile(str(README))
                elif sys.platform == "darwin":
                    subprocess.run(["open", str(README)])
                else:
                    subprocess.run(["xdg-open", str(README)])
            else:
                messagebox.showerror("Error", f"README.md not found:\n{README}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


class RunnerWindow(tk.Toplevel):
    def __init__(self, master: App, use_cuda: bool):
        super().__init__(master)
        self.title("Pipeline Progress")
        self.geometry("920x600")
        self.resizable(True, True)
        self.use_cuda = use_cuda

        self.protocol("WM_DELETE_WINDOW", self.on_close_blocked)

        self.info = ttk.Label(self, text="Running pipeline…", font=("Segoe UI", 12, "bold"))
        self.info.pack(pady=(12, 4))

        self.progress = ttk.Progressbar(self, orient="horizontal", length=860, mode="determinate")
        self.progress.pack(pady=6)

        self.text = scrolledtext.ScrolledText(self, wrap="word", height=26, font=("Consolas", 10))
        self.text.pack(fill="both", expand=True, padx=10, pady=8)
        self.text.configure(state="disabled")

        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(pady=8)

        self.btn_open_results = ttk.Button(self.button_frame, text="Open results in Explorer", command=self.open_results)
        self.btn_back = ttk.Button(self.button_frame, text="Return to main menu", command=self.return_to_main)
        self.btn_open_results.grid_remove()
        self.btn_back.grid_remove()

        self.steps = pipeline_commands()
        self.progress["maximum"] = len(self.steps)
        self._stop = False

        t = threading.Thread(target=self.run_steps, daemon=True)
        t.start()

    def append(self, msg: str):
        self.text.configure(state="normal")
        self.text.insert("end", msg)
        self.text.see("end")
        self.text.configure(state="disabled")
        self.update_idletasks()

    def run_steps(self):
        base_env = os.environ.copy()
        base_env["PYTHONIOENCODING"] = "utf-8"
        if self.use_cuda:
            base_env["USE_CUDA"] = "1"

        for i, cmd in enumerate(self.steps, start=1):
            if self._stop:
                break
            self.append(f"\n$ {' '.join(cmd)}\n")

            try:
                with subprocess.Popen(
                    cmd,
                    cwd=str(REPOSITORY_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    universal_newlines=True,
                    env=base_env,
                ) as proc:
                    for line in proc.stdout or []:
                        self.append(line)
                    ret = proc.wait()
            except Exception as e:
                ret = -1
                self.append(f"\n[ERROR] {e}\n")

            if ret != 0:
                self.append(f"\n[FAILED] Step returned {ret}. Aborting pipeline.\n")
                messagebox.showerror("Pipeline failed", f"Command failed:\n{' '.join(cmd)}\n\nCheck the log above.")
                self.info.config(text="Pipeline failed.")
                return

            self.progress["value"] = i
            self.info.config(text=f"Completed {i}/{len(self.steps)}")

        self.info.config(text="All steps completed successfully.")
        self.btn_open_results.grid(row=0, column=0, padx=8)
        self.btn_back.grid(row=0, column=1, padx=8)

    def on_close_blocked(self):
        messagebox.showwarning("Busy", "Please wait until the pipeline finishes.")

    def return_to_main(self):
        self.destroy()

    def open_results(self):
        for p in RESULT_DIRECTORY:
            if p.exists():
                try:
                    if os.name == "nt":
                        os.startfile(str(p))
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", str(p)])
                    else:
                        subprocess.Popen(["xdg-open", str(p)])
                except Exception as e:
                    messagebox.showerror("Open error", f"Could not open {p}:\n{e}")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
