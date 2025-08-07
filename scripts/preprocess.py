import argparse, joblib, pandas as pd, sys
from pathlib import Path

# Add parent directory to path for src imports
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocess import build_preprocessing_pipeline
import src.config as cfg

def run(stage: str) -> None:
    # Ensure directories exist
    cfg.ensure_directories()
    
    csv_path = cfg.DATA_RAW / f"{stage}.csv"
    df = pd.read_csv(csv_path)

    pipe = build_preprocessing_pipeline()
    if stage == "train":
        X = pipe.fit_transform(df)
        joblib.dump(pipe, cfg.MODELS_DIR / "preproc.pkl")
    else:
        pipe = joblib.load(cfg.MODELS_DIR / "preproc.pkl")
        X = pipe.transform(df)

    out = cfg.DATA_PROCESSED / f"{stage}.parquet"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    
    # Get feature names with fallback (updated for enhanced features)
    try:
        feature_names = pipe.get_feature_names_out()
    except:
        num_features = [f"num__{feat}" for feat in cfg.KEEP_FEATURES]
        cat_features = [f"cat__{cfg.CAT_COL}_male"]  # Only one category since drop="first"
        feature_names = num_features + cat_features
    
    pd.DataFrame(X, columns=feature_names).to_parquet(out)
    print(f"[preprocess] {stage} → {out}, shape={X.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "test"], required=True)
    run(ap.parse_args().stage)