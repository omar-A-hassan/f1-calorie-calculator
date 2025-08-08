# F1 Racer Diet Planning: ML Pipeline for Caloric Expenditure Prediction

## Project Overview

A production-ready machine learning pipeline designed to predict caloric expenditure for Formula 1 racers based on physiological and environmental parameters. The system employs advanced feature engineering, ensemble learning, and automated deployment through GitHub Actions with DVC (Data Version Control) orchestration.


## Architecture

### High-Level System Design

```
Raw Data → Feature Engineering → Model Training → Stacking → Submission
    ↓              ↓                   ↓            ↓           ↓
  CSV Files    Physiological       4 Base Models  XGBoost    submission.csv
              Features (18)        (LGB,Cat,      Meta-
                                  XGB,Enet)      Learner
```

### Technology Stack

- **ML Framework**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Feature Engineering**: Custom physiological transformers
- **Orchestration**: DVC (Data Version Control)
- **CI/CD**: GitHub Actions
- **Environment**: Python 3.11, Conda
- **Optimization**: Optuna hyperparameter tuning

## Features

### Advanced Physiological Feature Engineering

The pipeline implements research-backed physiological features proven to correlate strongly with caloric expenditure:

- **Heart Rate Reserve (HRR)**: `(HR - 65) / (208 - 0.7*Age - 65) * 100`
- **Basal Metabolic Rate (BMR)**: Mifflin-St Jeor equations (sex-specific)
- **Training Impulse (TRIMP)**: Duration × (HRR/100) - time-intensity integration
- **MET-adjusted Energy**: Individual metabolic equivalent calculations
- **Advanced Interactions**: HR², HR×Duration, HR×Body_Temp, BMI×Duration

### Model Ensemble

- **Base Models**: 4 optimized regressors with enhanced hyperparameter spaces
- **Meta-Learning**: XGBoost stacking with 10 meta-features (statistical + advanced)
- **Cross-Validation**: Stratified by Sex and Duration bins for demographic balance
- **Optimization**: 20 trials per model with Optuna for sub-6-hour execution

### Infrastructure

- **Automated Pipeline**: Complete end-to-end execution via GitHub Actions
- **Version Control**: DVC tracks data, models, and pipeline state
- **Error Handling**: Comprehensive fallback mechanisms and validation
- **Monitoring**: Detailed logging and performance metrics

## Installation

### Prerequisites

- Python 3.11+
- Conda/Miniconda
- Git with DVC extension

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd calorie-competition

# Create conda environment
conda env create -f environment.yml
conda activate calories_py311

# Install DVC
pip install dvc

# Initialize DVC (if not already initialized)
dvc init
```

### Dependencies

Core dependencies managed through `environment.yml`:

```yaml
dependencies:
  - python=3.11
  - numpy>=1.26
  - pandas>=2.2
  - scikit-learn>=1.4
  - lightgbm=4.4.*
  - catboost=1.2.*
  - xgboost=2.0.*
  - optuna>=3.6
  - jupyterlab
```

## Usage

### Local Execution

```bash
# Run complete pipeline
dvc repro

# Run specific stages
dvc repro preprocess_train
dvc repro train_models
dvc repro stack_models

# Check pipeline status
dvc status
dvc dag
```

### Individual Components

```bash
# Preprocessing only
python scripts/preprocess.py --stage train
python scripts/preprocess.py --stage test

# Model training
python scripts/train.py --model lgb --trials 20
python scripts/train.py --model all --trials 20

# Stacking
python src/stacking.py

# Generate submission
python src/make_submission.py
```

## Pipeline Components

### 1. Data Preprocessing (`src/preprocess.py`)

**Winsorization**: Outlier handling with physiological bounds
```python
WINSOR_LIMITS = {
    "Heart_Rate": (74.0, 116.0),
    "Body_Temp": (37.7, 41.2),
    "HRR": (0.0, 150.0),
    "BMR": (800.0, 2500.0)
}
```

**Feature Engineering**: Custom transformer creating 11 derived features
- Core physiological metrics (HRR, BMR, TRIMP)
- Statistical interactions (BMI×Duration, HR×Body_Temp)
- Non-linear transformations (HR²)

**Output**: 18 features (6 raw + 11 derived + 1 categorical)

### 2. Model Training (`src/train_models.py`)

**Enhanced Cross-Validation**:
- Stratified by Sex and Duration quantiles (33rd, 67th percentiles)
- Ensures demographic balance across folds
- Consistent with physiological variance patterns

**Base Models with Optimized Hyperparameters**:

- **LightGBM**: Enhanced with `feature_fraction`, `reg_alpha`, `verbosity=-1`
- **CatBoost**: Advanced `grow_policy`, `bootstrap_type`, `border_count`
- **XGBoost**: L1/L2 regularization, `min_child_weight` optimization
- **ElasticNet**: `max_iter=5000` for convergence, increased regularization paths

### 3. Stacking (`src/stacking.py`)

**Meta-Feature Generation**:
```python
# Statistical meta-features
pred_mean, pred_std, pred_range, pred_median
# Advanced meta-features  
pred_var, pred_skew
```

**XGBoost Meta-Learner**:
```python
XGBRegressor(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    reg_alpha=0.1,
    reg_lambda=1.0
)
```

**Fallback Mechanism**: Automatic RidgeCV fallback if XGBoost fails

### 4. Submission Generation (`src/make_submission.py`)

**Enhanced Compatibility**:
- Recreates training meta-features for test data
- Preserves feature column ordering from training
- Comprehensive validation and error handling


## Development

### Project Structure

```
├── data/
│   ├── raw/                 # Original CSV files
│   └── processed/           # Processed parquet files
├── models/                  # Trained model artifacts
├── src/                     # Source code
│   ├── config.py           # Configuration constants
│   ├── preprocess.py       # Feature engineering pipeline
│   ├── train_models.py     # Model training with Optuna
│   ├── stacking.py         # Enhanced meta-learning
│   └── make_submission.py  # Submission generation
├── scripts/                # CLI wrappers
├── notebooks/              # Jupyter notebooks for EDA
├── .github/workflows/      # GitHub Actions CI/CD
├── dvc.yaml               # DVC pipeline definition
├── environment.yml        # Conda environment spec
└── README.md             # This file
```

### Configuration Management

Central configuration in `src/config.py`:
- **Feature Lists**: `RAW_NUM_FEATURES`, `DERIVED_FEATURES`, `KEEP_FEATURES`
- **Model Parameters**: `N_TRIALS`, `N_SPLITS`, `RANDOM_STATE`
- **Paths**: `PROJECT_ROOT`, `DATA_RAW`, `MODELS_DIR`
- **Preprocessing**: `WINSOR_LIMITS` for physiological bounds


## Performance Optimization

### Hyperparameter Optimization

**Optuna Integration**:
- **Objective**: Root Mean Squared Error minimization
- **Search Strategy**: Tree-structured Parzen Estimator (TPE)
- **Pruning**: Early stopping for poor-performing trials
- **Parallelization**: Single-threaded for reproducibility

**Search Spaces**:
- **LightGBM**: 9 parameters including feature selection
- **CatBoost**: 8 parameters with growth policies
- **XGBoost**: 8 parameters with regularization
- **ElasticNet**: 3 parameters with convergence tuning

