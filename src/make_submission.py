"""Predict test set with stacked model and create submission.csv."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

import config
from stacking import load_lvl1_preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    df_test = pd.read_feather(config.DATA_PROCESSED / "test.feather")
    lvl1_test = load_lvl1_preds(df_test)

    meta = joblib.load(config.MODELS_DIR / "stacked.pkl")
    preds = meta.predict(lvl1_test)

    sub = pd.DataFrame({"id": df_test["id"], "Calories": preds})
    sub.to_csv(args.output, index=False)
    print(f"[make_submission] Saved {args.output} with shape {sub.shape}")


if __name__ == "__main__":
    main()
