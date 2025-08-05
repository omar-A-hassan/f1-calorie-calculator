"""Train & tune four models with Optuna; save best of each."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

import config


NUM_COLS = config.RAW_NUM_FEATURES + config.DERIVED_FEATURES
CAT_COLS = [config.CAT_COL]


def objective_lgb(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 7, 255),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": config.RANDOM_STATE,
    }
    model = LGBMRegressor(**params)

    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    rmses = []
    for train_idx, val_idx in cv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])
        rmses.append(mean_squared_error(y.iloc[val_idx], pred, squared=False))
    return float(pd.Series(rmses).mean())


def run_optuna(X: pd.DataFrame, y: pd.Series, model_name: str, objective_func) -> Path:
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_func(trial, X, y), n_trials=config.N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    if model_name == "lgb":
        final_model = LGBMRegressor(**best_params)
    elif model_name == "enet":
        final_model = ElasticNet(**best_params)  # type: ignore[arg-type]
    else:
        raise NotImplementedError(model_name)

    final_model.fit(X, y)
    model_path = config.MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(final_model, model_path)
    print(f"[train_models] Saved {model_name} to {model_path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgb", "enet"], required=True)
    args = parser.parse_args()

    df = pd.read_feather(config.DATA_PROCESSED / "train.feather")
    y = df.pop(config.TARGET_COL)
    X = df[NUM_COLS + CAT_COLS]

    if args.model == "lgb":
        run_optuna(X, y, "lgb", objective_lgb)
    elif args.model == "enet":
        # Simple ElasticNet sweep inline
        def objective_enet(trial: optuna.Trial, X: pd.DataFrame, y):
            params = {
                "alpha": trial.suggest_float("alpha", 0.001, 1.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            }
            model = ElasticNet(**params, random_state=config.RANDOM_STATE)
            cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
            rmses = []
            for tr, va in cv.split(X):
                model.fit(X.iloc[tr], y.iloc[tr])
                pred = model.predict(X.iloc[va])
                rmses.append(mean_squared_error(y.iloc[va], pred, squared=False))
            return float(pd.Series(rmses).mean())

        run_optuna(X, y, "enet", objective_enet)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
