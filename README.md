<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
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
      <a href="#about-the-project">About The Project</a>
      <a href="#built-with">Built With</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
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

3.  Transformer model evaluation

        Evaluates pre-trained transformer (zero-shot) models without fine-tuning:
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
- A systematic comparison of shallow and transformer models on diverse datasets.
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
| GPU              | Nvidia RTX 2060/Intel Iris Xe (avoid AMD) |     Nvidia RTX 3060+     |
| CUDA toolkit     |               Not mandatory               |        CUDA 12.1         |
| Disk space       |                    2GB                    |           8GB            |





1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/github_username/repo_name/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=github_username/repo_name" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
