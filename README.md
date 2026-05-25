# Guidence

This content is divided into two parts: one focuses on reproducing TrialDura, and the detailed implementation steps can be found in the TrialDura GitHub repository.

The other part further explores machine learning methods based on selected components of TrialDura, such as feature selection and the training/testing set split strategy.

# Machine Learning Exploration for Clinical Trial Duration Prediction

This folder contains the machine learning exploration component of the clinical trial duration prediction project.

Unlike the **Reproduction** section, which focuses on reproducing the TrialDura framework, this section explores alternative machine learning approaches inspired by parts of the TrialDura pipeline, including feature selection, train/test split strategy, and clinical trial text preprocessing.

## Project Goal

The objective of this exploration is to investigate whether traditional machine learning models and lightweight neural architectures can effectively predict clinical trial duration using structured and text-derived features from ClinicalTrials.gov data.

The target variable is:

- **Clinical trial duration (days)**  
  Calculated as:

```text
primary_completion_date - start_date
```

---

## Features Used

The experiments primarily use text-based features extracted from XML trial records:

- Trial title
- Brief summary
- Inclusion criteria
- Exclusion criteria
- Disease / condition
- Drug intervention

Additional structured features explored in some experiments:

- Trial phase
- Enrollment

Text features are embedded using:

- **Bio_ClinicalBERT**

Embedding strategy:

- Mean pooling over token embeddings
- Maximum token length: 512

---

## Models Explored

This exploration includes the following models:

### Traditional Machine Learning Models

- Linear Regression
- Random Forest Regressor
- AdaBoost Regressor
- XGBoost Regressor
- Multi-Layer Perceptron (MLP)

### Neural Architectures

- MLP + Transformer
- Hyperparameter-tuned Transformer + MLP variants

---

## Experiment Overview

### 1. Baseline Machine Learning Models

**File:** `train_ml_models_hpc.py`

Trains traditional machine learning models using BioClinicalBERT embeddings generated from selected trial features.

Purpose:

- Establish baseline performance
- Compare traditional models against neural architectures

---

### 2. Feature Embedding Generation

**File:** `embedding_selected_features_hpc.py`

Extracts selected XML features and generates BioClinicalBERT embeddings.

Outputs cached embeddings for downstream experiments.

---

### 3. Feature Ablation Study

**File:** `feature_ablation_models_hpc.py`

Evaluates feature importance by removing specific input features.

Ablation settings include:

- all_features
- no_inclusion_criteria
- no_exclusion_criteria
- no_criteria
- no_drug
- no_disease
- no_title
- no_summary

Purpose:

- Understand feature contribution
- Identify the most informative trial metadata

---

### 4. Log Transformation Analysis

**File:** `log_models_compare.py`

Applies log transformation to the target variable:

```python
log1p(duration)
```

Then transforms predictions back using:

```python
expm1(prediction)
```

Purpose:

- Address right-skewed duration distribution
- Evaluate whether transformation improves prediction performance

---

### 5. Phase-Specific Analysis

**File:** `mlp_phase_analysis.py`

Analyzes model performance across different clinical trial phases.

Purpose:

- Examine phase-dependent prediction difficulty
- Investigate whether separate models may perform better for specific phases

---

### 6. Transformer + MLP Model

**File:** `train_transformer_mlp_hpc.py`

Applies a Transformer encoder over feature embeddings before regression.

Purpose:

- Capture relationships between trial features
- Compare against standard MLP baselines

---

### 7. Transformer Hyperparameter Search

**File:** `train_transformer_mlp_hparam_phase_enrollment_hpc.py`

Performs hyperparameter experiments for Transformer + MLP models.

Explored parameters:

- hidden dimension
- attention heads
- transformer layers
- dropout
- learning rate
- weight decay
- MLP hidden size

---

### 8. Outlier Analysis

**File:** `train_transformer_mlp_original_vs_no_extreme.py`

Compares model performance:

- with original data
- after removing extreme duration outliers

Purpose:

- Evaluate outlier impact on model stability

---

### 9. Visualization

**File:** `Visualization.Rmd`

Contains exploratory visualizations and performance comparison plots.

Includes:

- duration distribution
- outlier analysis
- model comparison
- ablation visualizations
- phase analysis

---

## Evaluation Metrics

Models are evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R²
- Pearson correlation
- Spearman correlation

---

## Computing Environment

Experiments were designed for HPC execution.

Typical environment:

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- XGBoost
- R (visualization)

---

## Notes

- This folder focuses on exploratory machine learning experiments rather than exact reproduction.
- The full TrialDura reproduction pipeline is documented separately in the `Reproduction` folder.
- Large datasets, embeddings, and model checkpoints are not included in this repository.

```


这样很规范。
