# Cancer Sample Type Classification Using Genomic and Clinical Data

## Overview

This project focuses on predicting **cancer sample types** (e.g., *Primary Tumor*, *First Recurrence*, *n Recurrence*) using **genomic mutation profiles** and **clinical features**. Accurate classification of cancer sample stages can assist in early intervention planning, treatment strategy, and deeper understanding of disease progression.

We use a combination of classical machine learning models, feature engineering, and interpretability tools (like SHAP) to build explainable models and identify key biomarkers.

---

## Project Goals

- **Classify sample types** based on genomic and clinical data.
- Understand **which genes** contribute most to classification.
- Assess the **trade-off between model performance and interpretability**.
- Lay the foundation for **biologically meaningful discoveries** using machine learning.

---

## Dataset Description

- **File**: `combined_data.tsv.txt`
- **Samples**: Cancer patient samples from various stages and recurrence types.
- **Features**:
  - Genomic mutation data (binary mutation status per gene).
  - Clinical metadata (e.g., cancer type, sample type, survival info).
  - Sample Type (our prediction target).

---

## What We Did 

### 1. Data Integration & Preparation

- Merged **genetic mutation data** with **reduced clinical metadata**, including only the most informative clinical features (e.g., cancer type, sample type).
- Excluded samples with missing or ambiguous labels—particularly *Metastasis*—as they couldn't be clearly assigned to the classification task.
- Reduced the target labels (`Sample Type`) from a larger set to **three consolidated categories**:
  - `Tumor Primary`
  - `First Recurrence`
  - `n Recurrence` (combined 2nd, 3rd, and 4th recurrence samples)
  
Reducing the classes improved clarity, eliminated sparsely populated categories, and enabled more stable model training.

---

### 2. Feature Engineering

- **Gene Mutations** were binarized:
  - `"WT"` (Wild Type) → `0`
  - Any mutation → `1`
- **Cancer Type** was one-hot encoded to retain categorical information without imposing ordinality.
- The **target variable** (`Sample Type`) was label-encoded.


Binary mutation encoding simplifies modeling and aligns with common practice in genomic machine learning. One-hot encoding preserves non-hierarchical categorical relationships.

---

### 3. Model Training & Comparison

Using stratified train/test splits (80/20), we trained and benchmarked the following models:

- **Random Forest**
- **Decision Tree**
- **Gradient Boosting**
- **XGBoost**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**

Metrics used:
- Accuracy
- Precision
- Recall
- F1 Score

Visualization:
- Confusion matrices
- Metric bar plots

A broad set of classifiers allowed us to test performance across different modeling assumptions—e.g., tree-based (non-linear), linear, and probabilistic.

---

### Feature Importance Analysis with SHAP

To interpret model decisions and understand **which genes influenced predictions**, we used **SHAP (SHapley Additive exPlanations)** on Decision Trees and Gradient Boosting models.

We created:
- **Class-specific SHAP summary plots** (each sample type colored separately)
- **Stacked bar charts** to compare gene contributions across all classes
- **Top gene lists** (20, 50, 100) based on mean absolute SHAP values

  
SHAP makes model decisions transparent and provides a pathway for identifying potential **biomarkers** relevant to recurrence.

---

### 5. Dimensionality Reduction for Modeling

We identified the **top 20 most important genes** based on SHAP values and re-trained models using only these features.


This reduced model retained much of the original performance while being easier to interpret, faster to train, and more practical for clinical use.

---

### 6. Clustering Exploration

As an unsupervised extension:
- Performed **K-Means clustering** on genetic profiles
- Used **Elbow** and **Silhouette plots** to determine the ideal number of clusters
- Proposed plotting:
  - X-axis: Sample Type
  - Y-axis: Number of mutated genes per sample

 
Clustering helps validate whether sample types naturally separate based on mutations and may uncover **latent subtypes** or structural groupings not captured by labels.

---

### 7. Model Tuning & Refinement

- Excluded *Metastasis* samples
- Focused on top-performing classifiers: **Gradient Boosting** and **Decision Tree**
- Fine-tuned models using reduced feature sets
- Aligned SHAP visualization strategy with academic examples (e.g., [BMC Medical Informatics](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-021-01420-1))

The final models are **both performant and interpretable**, which is crucial for clinical applications. Fine-tuning on fewer features also improves robustness.

---





