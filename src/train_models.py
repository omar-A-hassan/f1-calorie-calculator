"""Train & tune four models with Optuna; save best of each."""
from __future__ import annotations

import argparse, joblib, datetime as dt
from pathlib import Path
import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline

try:
    from . import config
except ImportError:
    import config


def create_enhanced_cv_strategy(df_raw: pd.DataFrame, n_splits: int = 5):
    """
    Create enhanced CV strategy with stratification by Sex and Duration bins.
    
    Returns:
        CV splitter object for use in cross-validation
    """
    # Create stratification variable combining Sex and Duration bins
    duration_quantiles = df_raw['Duration'].quantile([0.33, 0.67]).values
    duration_bins = pd.cut(df_raw['Duration'], 
                          bins=[-np.inf, duration_quantiles[0], duration_quantiles[1], np.inf],
                          labels=['short', 'medium', 'long'])
    
    # Combine Sex and Duration bins for stratification
    strat_variable = df_raw['Sex'].astype(str) + '_' + duration_bins.astype(str)
    
    # Create StratifiedKFold with combined stratification variable
    cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                                 random_state=config.RANDOM_STATE)
    
    return cv_splitter, strat_variable


def load_training_frame() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Returns X (processed features), y (target), and df_raw for CV strategy.
    - X: Enhanced feature matrix from data/processed/train.parquet
    - y: Calories column merged from raw CSV  
    - df_raw: Raw training data for CV stratification
    """
    X = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")
    df_raw = pd.read_csv(config.DATA_RAW / "train.csv").set_index("id")
    
    y = df_raw.loc[X.index, config.TARGET_COL]
    
    # ensure correct column order from fitted preproc
    preproc = joblib.load(config.MODELS_DIR / "preproc.pkl")
    try:
        feat_names = list(preproc.get_feature_names_out())
    except:
        # Fallback if get_feature_names_out fails (updated for enhanced features)
        num_features = [f"num__{feat}" for feat in config.KEEP_FEATURES]
        cat_features = [f"cat__{config.CAT_COL}_male"]  # Only one category since drop="first"
        feat_names = num_features + cat_features
    X = X[feat_names]
    return X, y, df_raw


def objective_lgb(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, 
                  cv_splitter, strat_variable: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 7, 255),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "verbosity": -1,  # Silence warnings
        "random_state": config.RANDOM_STATE,
    }
    model = LGBMRegressor(**params)

    rmses = []
    for train_idx, val_idx in cv_splitter.split(X, strat_variable):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])
        rmses.append(root_mean_squared_error(y.iloc[val_idx], pred))
    return float(pd.Series(rmses).mean())


def objective_cat(trial, X, y, cv_splitter, strat_variable: pd.Series):
    params = {
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "iterations": trial.suggest_int("iterations", 300, 1200, step=150),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "RMSE",
        "random_seed": config.RANDOM_STATE,
        "verbose": 0,
    }
    model = CatBoostRegressor(**params)
    rmses = []
    for tr, va in cv_splitter.split(X, strat_variable):
        model.fit(X.iloc[tr], y.iloc[tr])
        rmses.append(root_mean_squared_error(y.iloc[va], model.predict(X.iloc[va])))
    return float(pd.Series(rmses).mean())


def objective_xgb(trial, X, y, cv_splitter, strat_variable: pd.Series):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": config.RANDOM_STATE,
        "objective": "reg:squarederror",
        "tree_method": "hist",
    }
    model = XGBRegressor(**params, n_jobs=4)
    rmses = []
    for tr, va in cv_splitter.split(X, strat_variable):
        model.fit(X.iloc[tr], y.iloc[tr])
        rmses.append(root_mean_squared_error(y.iloc[va], model.predict(X.iloc[va])))
    return float(pd.Series(rmses).mean())


def run_optuna(X: pd.DataFrame, y: pd.Series, df_raw: pd.DataFrame, 
               model_name: str, objective_func) -> Path:
    # Create enhanced CV strategy
    cv_splitter, strat_variable = create_enhanced_cv_strategy(df_raw, config.N_SPLITS)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_func(trial, X, y, cv_splitter, strat_variable), 
                  n_trials=config.N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    if model_name == "lgb":
        final_model = LGBMRegressor(**best_params, verbosity=-1)  # Add verbosity here too
    elif model_name == "enet":
        final_model = ElasticNet(**best_params, max_iter=5000)  # Increase to fix convergence
    elif model_name == "cat":
        final_model = CatBoostRegressor(**best_params, loss_function="RMSE",
                                        random_seed=config.RANDOM_STATE, verbose=0)
    elif model_name == "xgb":
        final_model = XGBRegressor(**best_params, objective="reg:squarederror",
                                   tree_method="hist", n_jobs=4)
    else:
        raise NotImplementedError(model_name)

    final_model.fit(X, y)
    model_path = config.MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(final_model, model_path)
    print(f"[train_models] Saved {model_name} to {model_path}")
    return model_path


def objective_enet(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, 
                   cv_splitter, strat_variable: pd.Series) -> float:
    params = {
        "alpha": trial.suggest_float("alpha", 0.001, 1.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "max_iter": 5000,  # Increase from 2000 to fix convergence issues with large duality gaps
        "random_state": config.RANDOM_STATE,
    }
    model = ElasticNet(**params)
    rmses = []
    for tr, va in cv_splitter.split(X, strat_variable):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        rmses.append(root_mean_squared_error(y.iloc[va], pred))
    return float(pd.Series(rmses).mean())


# Load training data once
X, y, df_raw = load_training_frame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgb", "enet", "cat", "xgb"], required=True)
    parser.add_argument("--trials", type=int, default=config.N_TRIALS)
    parser.add_argument("--folds", type=int, default=config.N_SPLITS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args()
    config.N_TRIALS = args.trials   # override globals
    config.N_SPLITS = args.folds
    config.RANDOM_STATE = args.seed

    print(f"[train_models] Enhanced pipeline with {X.shape[1]} features")
    print(f"[train_models] Using stratified CV by Sex and Duration bins")
    print(f"[train_models] Training {args.model} with {args.trials} trials")

    if args.model == "lgb":
        run_optuna(X, y, df_raw, "lgb", objective_lgb)
    elif args.model == "enet":
        run_optuna(X, y, df_raw, "enet", objective_enet)
    elif args.model == "cat":
        run_optuna(X, y, df_raw, "cat", objective_cat)
    elif args.model == "xgb":
        run_optuna(X, y, df_raw, "xgb", objective_xgb)


if __name__ == "__main__":
    main()