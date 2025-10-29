<a id="readme-top"></a>
<!-- PROJECT SHIELDS -->
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]
[![project_license][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/marvinalnashi/Requirements-Classification-Replication-Package">
    <img src="images/logo.png" alt="Logo" width="80" height="100">
  </a>

<h3 align="center">Requirements classification</h3>

  <p align="center">
    Software Requirements Classification with Classical ML and Transformers: The Role of Dataset Quality and Generalisation
  </p>
<br>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#why-this-work-matters">Why this work matters</a>
    </li>
    <li>
      <a href="#research-motivation">Research motivation</a>
    </li>
    <li>
      <a href="#purpose-of-the-replication-package">Purpose of the replication package</a>
    </li>
    <li>
      <a href="#research-questions">Research questions</a>
    </li>
    <li>
      <a href="#experimental-framework-and-design">Experimental framework and design</a>
    </li>
    <li>
      <a href="#contribution">Contribution</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#interpreting-the-results">Interpreting the results</a>
      <ul>
        <li><a href="#figures">Figures</a></li>
        <li><a href="#numerical-outputs">Numerical outputs</a></li>
        <li><a href="#how-the-results-connect-to-the-research-questions">How the results connect to the research questions</a></li>
        <li><a href="#recommended-next-steps-for-researchers">Recommended next steps for researchers</a></li>
      </ul>
    </li>
    <li>
      <a href="#citations-and-references">Citations and references</a>
    </li>
  </ol>
</details>

[![Product Name Screen Shot][product-screenshot]](https://example.com)

<!-- ABOUT THE PROJECT -->
## Introduction

```
Requirements-Classification-Replication-Package/
├── data/
│   ├── PROMISE-Relabelled/
│   │   ├── promise-reclass.csv                    # Original PROMISE-Relabelled dataset
│   │   ├── processed/
│   │   │   ├── test.csv                           # Testing subset (after preprocessing)
│   │   │   └── train.csv                          # Training subset (after preprocessing)
│   └── PURE/
│       ├── Pure_Annotate_Dataset.csv              # Original PURE dataset
│       └── processed/
│           ├── test.csv                           # Testing subset (after preprocessing)
│           └── train.csv                          # Training subset (after preprocessing)
├── requirements.txt                               # List of required Python packages
├── results/
│   ├── figures/                                   # All generated visualizations
│   │   ├── bartmnli_promise_confusion.png
│   │   ├── bartmnli_pure_confusion.png
│   │   ├── logreg_promise_confusion.png
│   │   ├── logreg_pure_confusion.png
│   │   ├── sbert_promise_confusion.png
│   │   ├── sbert_pure_confusion.png
│   │   ├── svm_promise_confusion.png
│   │   └── svm_pure_confusion.png
│   ├── models/                                    # Saved .pkl models (SVM, Logistic Regression)
│   │   ├── logreg_promise.pkl
│   │   ├── logreg_pure.pkl
│   │   ├── svm_promise.pkl
│   │   └── svm_pure.pkl
│   ├── stats/                                     # Saved evaluation metrics in JSON format
│   │   ├── bartmnli_promise_metrics.json
│   │   ├── bartmnli_pure_metrics.json
│   │   ├── logreg_promise_metrics.json
│   │   ├── logreg_pure_metrics.json
│   │   ├── sbert_promise_metrics.json
│   │   ├── sbert_pure_metrics.json
│   │   ├── svm_promise_metrics.json
│   │   └── svm_pure_metrics.json
│   └── tables/                                    # Predictions and result tables in CSV format
│       ├── bartmnli_promise_predictions.csv
│       ├── bartmnli_pure_predictions.csv
│       ├── logreg_promise_promise_predictions.csv
│       ├── logreg_pure_pure_predictions.csv
│       ├── sbert_promise_predictions.csv
│       ├── sbert_pure_predictions.csv
│       ├── svm_promise_promise_predictions.csv
│       └── svm_pure_pure_predictions.csv
├── src/
│   ├── clean_generated.py                         # Cleans all generated files (models, results, tables)
│   ├── evaluation/
│   │   ├── evaluate_predictions.py                # Computes accuracy, F1, and JSON stats
│   │   └── generate_all_figures.py                # Generates confusion matrices and comparisons
│   ├── preprocessing.py                           # Cleans and prepares datasets (PROMISE & PURE)
│   ├── shallow-model/
│   │   ├── predict_shallow.py                     # Generates predictions from trained shallow models
│   │   ├── train_logreg.py                        # Trains Logistic Regression classifier
│   │   └── train_svm.py                           # Trains Support Vector Machine classifier
│   └── transformer-based-model/
│       ├── evaluate_bart_mnli.py                  # Zero-shot inference using BART-large-MNLI
│       └── evaluate_sbert.py                      # Zero-shot similarity-based inference using SBERT
└── start.py                                       # GUI launcher for the application
```
Automatic classification of software requirements is a central problem in Requirements Engineering and Software Engineering research.
Requirements define what a system should do (Functional Requirements, FRs) and how it should behave (Non-Functional Requirements, NFRs).
Correctly distinguishing between these two types of requirements is crucial, as misclassification can lead to misunderstandings, design flaws, and costly project failures.

However, manual requirement classification is a slow, error-prone, and often inconsistent process. To overcome this limitation, the research community has turned to machine learning and natural language processing methods.
While early approaches relied on shallow models such as Logistic Regression and Support Vector Machines trained on TF-IDF features, more recent research explores transformer-based language models such as BERT, RoBERTa, or NoRBERT, which learn deep contextual representations of text.

This repository provides a replication package to support the experimental framework I have developed for the associated study.
The paper investigates how dataset design, model type, and evaluation strategy affect classification performance and model generalisation in software requirements classification.
The replication package allows anyone to reproduce, inspect, and extend these experiments with minimal effort through an intuitive graphical interface.

Here's a blank template to get started. To avoid retyping too much info, do a search and replace with your text editor for the following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`, `project_license`


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Why this work matters

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

## Research motivation

The motivation for this study arises from three persistent challenges in the field:

Dataset Quality and Consistency
- The original PROMISE dataset suffered from inconsistent labels and limited annotation detail.
Its Relabelled PROMISE version (Dalpiaz et al., 2019) corrected these issues and introduced multi-label annotations, allowing requirements to be tagged as both FR and NFR when appropriate.
- The PURE dataset, derived from public SRS documents, uses a single-label policy (each requirement is either FR or NFR).
While simpler and cleaner, this approach reduces realism because many real-world requirements combine functional and qualitative aspects.

Generalisation Across Projects

- Most prior studies used random data splits within a single dataset, which can inflate results because similar requirements appear in both training and testing sets.
To test true robustness, models must be evaluated across datasets (for instance, train on PROMISE-Relabelled and test on PURE).

Evaluation Protocols
- Many studies use k-fold cross-validation without project awareness.
This work follows group-aware and cross-dataset evaluation, ensuring no project leakage and thus producing realistic generalisation metrics.

## Purpose of the replication package

The replication package automates the complete process of the associated study, from preprocessing to evaluation and visualisation.
It is designed to make the experiments fully reproducible, transparent, and accessible to researchers.

The application performs the following major steps automatically:

1.  Data Preprocessing
        
        Cleans, normalises, and splits the PROMISE-Relabelled and PURE datasets into training and testing subsets.

2.  Shallow model training

        Trains shallow machine-learning classifiers:
            Support Vector Machine
            Logistic Regression
            Both are applied separately to PROMISE and PURE.

3.  Transformer-based model evaluation

        Evaluates pre-trained transformer-based (zero-shot) models without fine-tuning:
            BART-large-MNLI for entailment-based classification
            SBERT for similarity-based classification

4.  Prediction and Evaluation
        
        Generates predictions for each dataset-model pair, computes metrics (accuracy, precision, recall, macro-/micro-F1), and exports confusion matrices. 

5.  Visualisation and Reporting

        Automatically produces high-quality figures, including:
        Confusion matrices per model and dataset
        Per-class F1 bar charts
        Summary tables and JSON-formatted metrics

6.  Reproducibility and Cleaning

        All generated files (models, stats, tables, and plots) are saved under /results.
        Users can reset the environment using clean_generated.py and rerun the full pipeline from scratch.


The start.py script provides a simple GUI that allows users to 
run the full experimental pipeline, 
enable or disable CUDA support, 
clean generated files, 
open this README documentation, and 
view results in the systems file manager application after completion.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Research questions

The study is structured around three central research questions:

| ID  | RQ | Focus |
|-----|----|-------|
| RQ1 | How do classical ML models compare with transformer-based models in requirements classification?   |  Model comparison and performance trade-offs     |
| RQ2 | What impact do different labelling schemes (mixed-label PROMISE vs. single-label PURE) have on classification accuracy and generalisation?   |   Dataset design and labelling effects    |
| RQ3 | How does performance change when models are trained on PROMISE and tested on PURE, and vice versa?   |   Cross-dataset robustness and transfer learning    |

These questions collectively explore how model choice, dataset structure, and evaluation design influence both performance and generalisability.

## Experimental framework and design

To address these questions, the study uses two methodological families:

1.  Classical/shallow models

      - Support Vector Machine
      - Logistic Regression

    These models use TF-IDF text representations and serve as interpretable baselines.
    They are trained and validated separately on PROMISE and PURE, allowing a controlled comparison.

2.  Transformer-based/zero-shot models

      - BART-large-MNLI, which treats requirement classification as a textual entailment task (“Does the requirement entail the label X?”). 
      - SBERT, which uses embedding similarity to associate requirements with labels.
    
    These models require no additional training and illustrate how pre-trained language models perform out-of-the-box on requirements engineering datasets.
    Other transformer-based models are purposefully avoided having to fine-tune them. Fine-tuning is a GPU-intensive process that requires high-end hardware.
    Therefore, it has been decided to circumvent it.

The evaluation process is both quantitative and visual:

- Metrics recorded: accuracy, precision, recall, macro-F1, micro-F1, per-class F1, and confusion matrices.
- Cross-dataset experiments: each model trained on PROMISE is tested on PURE and vice versa.
- Figures and tables: visualisations are generated automatically by evaluate_predictions.py and generate_all_figures.py.
- Outputs: stored under /results/ as CSV tables, JSON metrics, and high-resolution plots.

This design ensures replicability, comparability, and statistical robustness.

## Contribution

This replication package offers five key contributions to the research community:

- A fully reproducible experimental pipeline for requirements classification.
- A systematic comparison of shallow and transformer-based models on diverse datasets.
- An empirical assessment of dataset quality and labelling effects on performance.
- A transparent evaluation of model generalisation using cross-dataset testing.
- A user-friendly application with a GUI that runs all experiments automatically.

Together, these contributions advance the methodological maturity of NLP4RE research by standardising how dataset quality, modelling choices, and generalisation are analysed.

## Installation

The replication package is fully implemented in Python 3.11 and supports both CPU and GPU (CUDA) execution.
Before running the application, ensure your system meets the following requirements.

| Component        |            Minimum requirement            | Recommended if using GPU |
|:-----------------|:-----------------------------------------:|:------------------------:|
| Python version   |               3.11 (64-bit)               |      3.11 (64-bit)       |
| Operating system |         Windows 11 (64-bit)/Linux         |   Windows 11 (64-bit)    |
| RAM              |                    8GB                    |          16GB+           |
| CPU              |            Intel Core i5 gen 8            |   Intel Core i5 gen 8+   |
| GPU              | NVIDIA RTX 2060/Intel Iris Xe (avoid AMD) |     NVIDIA RTX 3060+     |
| CUDA toolkit     |               Not mandatory               |        CUDA 12.1         |
| Disk space       |                    2GB                    |           8GB            |

This section describes how to clone the repository, configure the environment, and run the replication package either through the graphical user interface (GUI) or by executing the Python scripts manually.
The goal is to enable full reproducibility of the experiments presented in the study, including all preprocessing, training, inference, evaluation, and visualisation steps.

1.  **Cloning the repository**

    Open a terminal and run: 
    ```sh
    git clone https://github.com/marvinalnashi/Requirements-Classification-Replication-Package.git
    cd Requirements-Classification-Replication-Package
    ```
    This will create a local copy of the repository containing the entire experimental framework.


2.  **Creating the Python virtual environment**

    The project uses Python 3.11. Using a clean virtual environment ensures compatibility and isolation from system packages.
    ```sh
    python -m venv venv311
    ```
    Activate the environment: 
    - Windows (PowerShell):
      ```sh
      venv311\Scripts\activate
      ```
    - Linux/macOS
      ```sh
      source venv311/bin/activate
      ```
      You should now see (venv311) at the start of your terminal.


3.  **Installing dependencies**

    All required packages are listed in requirements.txt. Install them using:
    ```sh
    pip install -r requirements.txt
    ```
    This will install core libraries that the application depends on.
    If you have an NVIDIA GPU and wish to use CUDA acceleration, ensure that your installed PyTorch build supports CUDA 12.1 or higher:
    ```sh
    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
    ```
    If CUDA is available, the GUI will automatically detect and use it. Otherwise, the application will fall back to CPU execution.
    To install the CUDA wheels for your Python version in your virtual environment, it is essential to determine which version of the CUDA stack to install.
    Please check whether your GPU is compatible with CUDA. Most NVIDIA GPUs are compatible but the version of the CUDA wheels to install in your virtual environment strongly depends on the specific graphics card used.
    [Visit](https://developer.nvidia.com/cuda-gpus) the CUDA GPU Compute Capability table to check this for your used GPU.
    To install the CUDA 12.4 stack from the official PyTorch wheel index, run the following: 
      ```sh
      pip install --upgrade pip
      pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
      ```

4.  **Running the experiments through the GUI (Choose either this step or step 5)**

    The easiest way to execute the full experimental pipeline is through the GUI provided in start.py.
    Run the application with:
    ```sh
    python start.py
    ```
    The interface provides four intuitive buttons and a checkbox in the main menu.
    - Run full pipeline
      1. Data preprocessing
      2. Shallow model training
      3. Transformer-based model evaluation
      4. Metric calculation and visualisation
      5. Report generation
    - Clean generated files: deletes all files that were generated by the application and allows the user to run the evaluation from scratch.
    - Open README.md: opens this README.md file in the text editor that is set to default in the system.
    - Exit: gracefully stop the start.py script and close the application.
    - CUDA checkbox: switch between GPU-accelerated and CPU-only execution. 
      Using a suitable GPU may drastically improve the performance of the pipeline.
    
    Clicking the button that runs the full pipeline opens a new window in which all the scripts of the application are run except for the script that removes the files the application generates.
    All scripts are run sequentially. A bar shows the current progress of the pipeline. Under the progress bar, a box displays the commands and script outputs as if the scripts were run inside a terminal. 
    After all scripts have completed and all artifacts have been generated, two buttons appear in the window.
    - Open results in Explorer: opens the results directory in the default file manager application of the system.
    - Return to main menu: closes the window and loads the main menu.
    

5.  **Running the experiments through the scripts manually (Choose either this step or step 4)**

    Advanced users may prefer to execute individual modules directly from the command line.
    Below is the recommended execution order and what each step takes as input and produces as output:
    - **Step 1: preprocessing.py**
      
        - Description: cleans/prepares datasets and performs feature extraction. Text is cleaned by removing URLs, enforcing lowercase, removing punctuations, and normalising whitespaces.
        - Input: Original dataset .csv files
        - Output: Processed dataset .csv files

    - **Step 2: train_logreg.py**
      
        - Description: Trains a Logistic Regression classifier using TF-IDF features.
        - Input: Preprocessed train/test .csv files
        - Output: Trained shallow models that are saved as .pkl (pickle) files

    - **Step 3: train_svm.py**
      
        - Description: Trains a Support Vector Machine classifier using TF-IDF features.
        - Input: Preprocessed train/test .csv files
        - Output: Trained shallow models that are saved as .pkl (pickle) files

    - **Step 4: predict_shallow.py**
      
        - Description: Loads a trained shallow model and writes predictions CSVs that match the transformer outputs.
        - Input: Trained shallow models that are saved as .pkl (pickle) files
        - Output: Prediction tables that are saved as .csv files

    - **Step 5: evaluate_bart_mnli.py**
      
        - Description: Runs BART-large-MNLI zero-shot inference for FR/NFR classification and NFR subclassification.
        - Input: Raw text from test .csv files
        - Output: Prediction tables that are saved as .csv files

    - **Step 6: evaluate_sbert.py**
      
        - Description: Runs SBERT similarity-based zero-shot classification.
        - Input: Raw text from test .csv files
        - Output: Prediction tables that are saved as .csv files

    - **Step 7: evaluate_predictions.py**
      
        - Description: Calculates the metrics accuracy, precision, recall, macro-F1, micro-F1, and per-class F1.
        - Input: Prediction tables that are saved as .csv files
        - Output: Metrics that are saved as .json files and confusion matrix figures.

    - **Step 8: generate_all_figures.py**
      
        - Description: Generates summary and comparative visualisations for all research questions.
        - Input: Metrics that are saved as .json.
        - Output: Bar charts, slope charts, and other figures and summary tables.

    All intermediate figures and final figures are stored in the /results/ directory, organised by type (models, stats, tables, and figures).


6.  **Cleaning the environment**

    To reset the project to a clean state (removing all generated models, plots, and metrics):
    ```sh
    python src/clean_generated.py
    ```
    This script safely removes all generated files while preserving the datasets and source code, allowing rerunning the experiments from scratch.
    The script can also be run by clicking the Clean generated files button in the main menu of the application's GUI.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Interpreting the results

Once the pipeline completes, the application generates a set of outputs located in the /results/ directory.
These include figures, metrics, trained models, and CSV summaries.
Each type of artifact corresponds to specific parts of the research and to one or more of the three research questions (RQs).

### Figures
1.  **Confusion matrices**

      - Location: /results/figures/*_confusion.png
      - Purpose: Show the true versus predicted labels for each model–dataset pair.
      - Interpretation:
          - The diagonal cells represent correctly classified requirements.
          - Off-diagonal cells indicate misclassifications.
          - A strong diagonal implies reliable classification performance.

2.  **Per-class F1-scores**

      - Location: /results/figures/*_perclass_f1.png
      - Purpose: Display how well each model predicts individual requirement categories.
      - Interpretation:
          - Useful for identifying classes that are underrepresented or difficult to distinguish.
          - Large variance between classes may indicate dataset imbalance.

3.  **Summary figures**

      - Location: /results/figures/*_summary.png
      - Purpose: Compare overall metrics (Macro-F1 and Accuracy/Micro-F1) per model.
      - Interpretation:
          - High Macro-F1 indicates good overall balance across classes.
          - The accuracy or micro-F1 metric reflects the proportion of correctly predicted samples.

4.  **Overall model comparison (RQ1)**

      - Location: /results/figures/overall_model_comparison.png
      - Purpose: Answers RQ1 – “How can traditional ML models be compared with transformer-based models in requirements classification?”
      - Interpretation:
          - SVM and Logistic Regression achieve the highest Macro-F1 on both datasets.
          - BART-MNLI and SBERT perform worse because they are zero-shot models not fine-tuned on PROMISE or PURE.
          - PROMISE scores are consistently higher than PURE due to dataset cleanliness and size.
      - Conclusion: 
          - Supervised shallow models outperform pre-finetuned but zero-shot transformer-based models in this setup. 
            If the transformer-based models were fine-tuned on PROMISE and PURE, the results may have been significantly different and may have aligned more with prior work.

5.  **Dataset impact (RQ2)**

      - Location: /results/figures/dataset_impact.png
      - Purpose: Answers RQ2 – “What impact do different labelling schemes have on classification accuracy and generalisation?”
      - Interpretation:
          - The slope chart shows how performance changes between PROMISE and PURE for each model.
          - All models experience a drop when moving from PROMISE (multi-label, consistent) to PURE (single-label, diverse).
          - The gap is largest for shallow models.
      - Conclusion: 
          - Dataset characteristics strongly influence model outcomes. 
            Cleaner, smaller datasets yield higher scores.

6.  **Cross-dataset generalisation (RQ3)**

      - Location: /results/figures/cross_dataset_generalisation_proxy.png
      - Purpose: Answers RQ3 – “How do classification accuracy and F1 score change when models trained on PROMISE are tested on PURE, and vice versa?”
      - Interpretation:
          - Shows ΔMacro-F1, the difference in performance between PROMISE and PURE.
          - Smaller bars indicate better stability and consistency across datasets.
          - Transformer-based models show more consistent (though lower) performance, while shallow models vary more strongly.
      - Conclusion: 
          - Transformers generalise more steadily across datasets, whereas shallow models perform better in-domain but degrade in cross-domain setups.

### Numerical outputs
1.  **Evaluation metrics**

    - Directory: /results/stats/
    - File type: .json
    - Description: Contains all evaluation metrics per model–dataset pair, including accuracy, precision, recall, macro-F1, micro-F1, per-class F1 scores.

2.  **Evaluation metrics**

    - Directory: /results/tables/
    - File type: .csv
    - Description: Contains prediction outputs and summary tables, useful for statistical comparison or replotting.

3.  **Evaluation metrics**

    - Directory: /results/models/
    - File type: .pkl
    - Description: Contains serialised versions of the trained shallow models. These can be reloaded for inference without retraining.

Each JSON file has the following structure: 
```sh
    {
      "accuracy": 0.842,
      "macro_f1": 0.793,
      "micro_f1": 0.851,
      "report": {
        "FR": {"precision": 0.85, "recall": 0.83, "f1-score": 0.84},
        "NFR": {"precision": 0.78, "recall": 0.76, "f1-score": 0.77},
        "macro avg": {"f1-score": 0.79},
        "weighted avg": {"f1-score": 0.81}
      }
    }
```

### How the results connect to the research questions
1.  **RQ1**

    - Figure: overall_model_comparison.png
    - Main output: Macro-F1 comparison across model families
    - Interpretation: Shallow models outperform transformers when trained directly on labeled data.

2.  **RQ2**

    - Figure: dataset_impact.png
    - Main output: Macro-F1 slopes from PROMISE to PURE
    - Interpretation: Dataset design significantly affects model accuracy. PURE introduces greater linguistic and topical variability.

3.  **RQ3**

    - Figure: cross_dataset_generalisation_proxy.png
    - Main output: ΔMacro-F1 differences
    - Interpretation: Transformers maintain more consistent performance across datasets, showing better generalisation stability.

### Recommended next steps for researchers
- Fine-tuning transformer-based models on PROMISE and PURE could close the gap with shallow models, yielding results that align more with prior work in this field.
- Expanding datasets with more balanced NFR categories would reduce per-class performance variance.
- Combining embedding-based and feature-based approaches (e.g., hybrid SVM + SBERT embeddings) may yield stronger cross-dataset results.
- Applying explainable AI (XAI) methods could reveal why specific requirement categories are harder to classify.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Citation and references
If you use this replication package or its derived outputs in your research or teaching, please cite it as follows: 

APA-style citation

    Al Nashi, M.H. (2025). Software Requirements Classification with Classical ML and Transformers: The Role of Dataset Quality and Generalisation (Replication Package).
    Utrecht University, Department of Information and Computing Sciences.
    GitHub repository: https://github.com/marvinalnashi/Requirements-Classification-Replication-Package

BibTeX entry

    @misc{alnashi2025requirements,
    author       = {Marvin Al Nashi},
    title        = {Software Requirements Classification with Classical ML and Transformers: The Role of Dataset Quality and Generalisation (Replication Package)},
    year         = {2025},
    institution  = {Utrecht University},
    howpublished = {\url{https://github.com/marvinalnashi/Requirements-Classification-Replication-Package}},
    note         = {Replication package accompanying a research study in Requirements Engineering and NLP and AI for Software Engineering}
    }


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/marvinalnashi/Requirements-Classification-Replication-Package.svg?style=for-the-badge
[contributors-url]: https://github.com/marvinalnashi/Requirements-Classification-Replication-Package/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/marvinalnashi/Requirements-Classification-Replication-Package.svg?style=for-the-badge
[forks-url]: https://github.com/marvinalnashi/Requirements-Classification-Replication-Package/network/members
[stars-shield]: https://img.shields.io/github/stars/marvinalnashi/Requirements-Classification-Replication-Package.svg?style=for-the-badge
[stars-url]: https://github.com/marvinalnashi/Requirements-Classification-Replication-Package/stargazers
[issues-shield]: https://img.shields.io/github/issues/marvinalnashi/Requirements-Classification-Replication-Package.svg?style=for-the-badge
[issues-url]: https://github.com/marvinalnashi/Requirements-Classification-Replication-Package/issues
[license-shield]: https://img.shields.io/github/license/marvinalnashi/Requirements-Classification-Replication-Package.svg?style=for-the-badge
[license-url]: https://github.com/marvinalnashi/Requirements-Classification-Replication-Package/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/marvinalnashi/
[product-screenshot]: images/screenshot.png
