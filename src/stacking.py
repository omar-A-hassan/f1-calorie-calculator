"""Stack first‑level model predictions into a meta‑learner."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import config


def load_lvl1_preds(df: pd.DataFrame) -> pd.DataFrame:
    """Generate level‑1 predictions as new features (one column per model)."""
    features = []
    for model_name in ["lgb", "enet"]:
        model: object = joblib.load(config.MODELS_DIR / f"{model_name}.pkl")
        preds = model.predict(df[config.RAW_NUM_FEATURES + config.DERIVED_FEATURES + [config.CAT_COL]])
        features.append(preds)
    return pd.DataFrame(np.vstack(features).T, columns=["pred_lgb", "pred_enet"])


def train_meta() -> None:
    df = pd.read_feather(config.DATA_PROCESSED / "train.feather")
    y = df.pop(config.TARGET_COL)
    lvl1 = load_lvl1_preds(df)

    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    rmses = []
    for tr, va in cv.split(lvl1):
        meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
        meta.fit(lvl1.iloc[tr], y.iloc[tr])
        pred = meta.predict(lvl1.iloc[va])
        rmses.append(mean_squared_error(y.iloc[va], pred, squared=False))
    print(f"[stacking] OOF RMSE = {np.mean(rmses):.4f}")
    joblib.dump(meta, config.MODELS_DIR / "stacked.pkl")


if __name__ == "__main__":
    train_meta()