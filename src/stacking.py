"""Stack first‑level model predictions into a meta‑learner."""
from pathlib import Path
import sys

# Add project root to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import KFold

try:
    from . import config
except ImportError:
    import config

BASE_MODELS = ["lgb", "cat", "xgb", "enet"]
PREPROC = joblib.load(config.MODELS_DIR / "preproc.pkl")
try:
    FEATURE_NAMES = list(PREPROC.get_feature_names_out())
except:
    # Fallback if get_feature_names_out fails (updated for enhanced features)
    num_features = [f"num__{feat}" for feat in config.KEEP_FEATURES]
    cat_features = [f"cat__{config.CAT_COL}_male"]  # Only one category since drop="first"
    FEATURE_NAMES = num_features + cat_features


def load_lvl1_preds(df: pd.DataFrame) -> pd.DataFrame:
    preds_matrix, names = [], []
    for m in BASE_MODELS:
        try:
            model = joblib.load(config.MODELS_DIR / f"{m}.pkl")
        except FileNotFoundError:
            continue
        preds_matrix.append(model.predict(df[FEATURE_NAMES]))
        names.append(f"pred_{m}")
    return pd.DataFrame(np.vstack(preds_matrix).T, columns=names)


def train_meta() -> None:
    X = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")[FEATURE_NAMES]
    y = (
        pd.read_csv(config.DATA_RAW / "train.csv", usecols=["id", config.TARGET_COL])
        .set_index("id")
        .loc[X.index, config.TARGET_COL]
    )
    lvl1 = load_lvl1_preds(X)

    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    rmses = []
    for tr, va in cv.split(lvl1):
        meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
        meta.fit(lvl1.iloc[tr], y.iloc[tr])
        pred = meta.predict(lvl1.iloc[va])
        rmses.append(root_mean_squared_error(y.iloc[va], pred))
    print(f"[stacking] OOF RMSE = {np.mean(rmses):.4f}")
    joblib.dump(meta, config.MODELS_DIR / "stacked.pkl")


if __name__ == "__main__":
    train_meta()