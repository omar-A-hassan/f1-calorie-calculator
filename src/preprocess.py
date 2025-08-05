"""Pre‑processing: winsorise tails, encode categorical, create features."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize  # type: ignore
from sklearn.preprocessing import StandardScaler

import config


def winsorise_df(df: pd.DataFrame, cols: list[str], limits: tuple[float, float] = (0.005, 0.005)) -> pd.DataFrame:  # noqa: E501
    df_out = df.copy()
    for c in cols:
        df_out[c] = winsorize(df_out[c], limits=limits)
    return df_out


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    h_m = df["Height (cm)"] / 100
    df["BMI"] = df["Weight (kg)"] / (h_m**2)
    df["HRxDur"] = df["Heart_Rate"] * df["Duration (min)"]
    df["Age_x_BodyTemp"] = df["Age"] * df["Body_Temp (C)"]
    df["HeatStressIdx"] = df["Body_Temp (C)"] * df["Heart_Rate"] / 100.0
    return df


def preprocess(csv_path: Path, is_train: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Winsorise numerical tails
    df = winsorise_df(df, config.RAW_NUM_FEATURES)

    # Encode Sex: male->1, female->0
    df[config.CAT_COL] = (df[config.CAT_COL].str.lower() == "male").astype(int)

    # Feature engineering
    df = add_domain_features(df)

    # Scale numeric features (fit on train, reuse params for test)
    num_cols = config.RAW_NUM_FEATURES + config.DERIVED_FEATURES
    scaler_path = config.MODELS_DIR / "scaler.pkl"
    scaler = StandardScaler()

    if is_train:
        df[num_cols] = scaler.fit_transform(df[num_cols])
        import joblib

        joblib.dump(scaler, scaler_path)
    else:
        import joblib

        scaler = joblib.load(scaler_path)
        df[num_cols] = scaler.transform(df[num_cols])

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Process train.csv when set; else test.csv")
    args = parser.parse_args()

    csv_name = "train.csv" if args.train else "test.csv"
    out_name = "train.feather" if args.train else "test.feather"

    df_processed = preprocess(config.DATA_RAW / csv_name, is_train=args.train)
    df_processed.reset_index(drop=True).to_feather(config.DATA_PROCESSED / out_name)
    print(f"[preprocess] Saved {out_name} with shape {df_processed.shape}")


if __name__ == "__main__":
    main()