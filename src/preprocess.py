"""sklearn Pipeline-based preprocessing with custom transformers."""
from __future__ import annotations

import argparse
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from . import config
except ImportError:
    import config


class Winsorizer(BaseEstimator, TransformerMixin):
    """Custom transformer for winsorizing features based on predefined limits."""
    
    def __init__(self, limits: Dict[str, Tuple[float, float]]):
        self.limits = limits
        
    def fit(self, X, y=None):
        """Fit does nothing - limits are predefined."""
        return self
        
    def transform(self, X):
        """Apply winsorization to specified features."""
        X_transformed = X.copy()
        
        for feature, (lower, upper) in self.limits.items():
            if feature in X_transformed.columns:
                X_transformed[feature] = X_transformed[feature].clip(lower=lower, upper=upper)
                
        return X_transformed
        
    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        return input_features


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for creating derived features."""
    
    def fit(self, X, y=None):
        """Fit does nothing - feature engineering is deterministic."""
        return self
        
    def transform(self, X):
        """Add BMI and HeatStressIdx features."""
        X_transformed = X.copy()
        
        # BMI = Weight / (Height_m²)
        height_m = X_transformed['Height'] / 100
        X_transformed['BMI'] = X_transformed['Weight'] / (height_m ** 2)
        
        # HeatStressIdx = Body_Temp × Heart_Rate / 100
        X_transformed['HeatStressIdx'] = (
            X_transformed['Body_Temp'] * X_transformed['Heart_Rate'] / 100
        )
        
        return X_transformed
        
    def get_feature_names_out(self, input_features=None):
        """Return feature names including new derived features."""
        if input_features is None:
            input_features = config.RAW_NUM_FEATURES + [config.CAT_COL]
        
        # Add derived features to input features
        output_features = list(input_features) + ['BMI', 'HeatStressIdx']
        return np.array(output_features)


def build_preprocessing_pipeline() -> Pipeline:
    """Build the complete preprocessing pipeline."""
    
    # Create the pipeline
    pipeline = Pipeline([
        ("winsor", Winsorizer(config.WINSOR_LIMITS)),
        ("feat", FeatureEngineer()),
        ("ct", ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), config.KEEP_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), [config.CAT_COL]),
            ],
            remainder="drop",
        ))
    ])
    
    return pipeline


def main() -> None:
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Preprocess data using sklearn Pipeline")
    parser.add_argument("--stage", choices=["train", "test"], required=True,
                       help="Stage to process: train or test")
    args = parser.parse_args()
    
    # Load data
    csv_path = config.DATA_RAW / f"{args.stage}.csv"
    df = pd.read_csv(csv_path)
    print(f"[preprocess] Loaded {csv_path}, shape={df.shape}")
    
    # Build pipeline
    pipe = build_preprocessing_pipeline()
    
    if args.stage == "train":
        # Fit and transform training data
        X_transformed = pipe.fit_transform(df)
        # Save the fitted pipeline
        pipeline_path = config.MODELS_DIR / "preproc.pkl"
        joblib.dump(pipe, pipeline_path)
        print(f"[preprocess] Pipeline fitted and saved to {pipeline_path}")
    else:
        # Load fitted pipeline and transform test data
        pipeline_path = config.MODELS_DIR / "preproc.pkl"
        pipe = joblib.load(pipeline_path)
        X_transformed = pipe.transform(df)
        print(f"[preprocess] Pipeline loaded from {pipeline_path}")
    
    # Convert to DataFrame with proper column names
    try:
        feature_names = pipe.get_feature_names_out()
    except:
        # Fallback: construct feature names manually
        num_features = [f"num__{feat}" for feat in config.KEEP_FEATURES]
        cat_features = [f"cat__{config.CAT_COL}_female", f"cat__{config.CAT_COL}_male"]
        feature_names = num_features + cat_features
    
    df_transformed = pd.DataFrame(X_transformed, columns=feature_names)
    
    # Save processed data
    output_path = config.DATA_PROCESSED / f"{args.stage}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_transformed.to_parquet(output_path, index=False)
    
    print(f"[preprocess] {args.stage} → {output_path}, shape={X_transformed.shape}")
    print(f"[preprocess] Columns: {list(df_transformed.columns)}")


if __name__ == "__main__":
    main()