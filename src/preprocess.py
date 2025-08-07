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
    """Enhanced transformer for creating research-backed physiological features."""
    
    def fit(self, X, y=None):
        """Fit does nothing - feature engineering is deterministic."""
        return self
        
    def transform(self, X):
        """Create comprehensive physiological and interaction features."""
        X_transformed = X.copy()
        
        # Core physiological features
        height_m = X_transformed['Height'] / 100
        X_transformed['BMI'] = X_transformed['Weight'] / (height_m ** 2)
        
        # Heart Rate Reserve (HRR) - most predictive feature
        # Formula: (HR - 65) / (208 - 0.7*Age - 65) * 100
        max_hr = 208 - 0.7 * X_transformed['Age']
        resting_hr = 65
        X_transformed['HRR'] = ((X_transformed['Heart_Rate'] - resting_hr) / 
                               (max_hr - resting_hr) * 100).clip(0, 150)
        
        # Basal Metabolic Rate (BMR) - sex-specific Mifflin-St Jeor equations
        bmr_base = (10 * X_transformed['Weight'] + 
                   6.25 * X_transformed['Height'] - 
                   5 * X_transformed['Age'])
        
        # Apply sex-specific adjustment
        is_male = (X_transformed[config.CAT_COL] == 'male')
        X_transformed['BMR'] = np.where(is_male, bmr_base + 5, bmr_base - 161)
        
        # Training Impulse (TRIMP) - Duration * (HRR/100)
        X_transformed['TRIMP'] = (X_transformed['Duration'] * 
                                 (X_transformed['HRR'] / 100)).clip(0, 50)
        
        # MET-adjusted energy expenditure
        # Approximate MET calculation based on HR intensity
        hrr_ratio = X_transformed['HRR'] / 100
        estimated_met = 3.5 + (hrr_ratio * 8)  # 3.5-11.5 MET range
        X_transformed['MET_Energy'] = (estimated_met * X_transformed['Weight'] * 
                                      X_transformed['Duration'] / 60 * 5).clip(10, 500)
        
        # Heat Stress Index
        X_transformed['HeatStressIdx'] = (
            X_transformed['Body_Temp'] * X_transformed['Heart_Rate'] / 100
        )
        
        # Advanced interaction features
        X_transformed['HR_Squared'] = X_transformed['Heart_Rate'] ** 2
        X_transformed['HR_Duration'] = X_transformed['Heart_Rate'] * X_transformed['Duration']
        X_transformed['HR_BodyTemp'] = X_transformed['Heart_Rate'] * X_transformed['Body_Temp']
        X_transformed['BMI_Duration'] = X_transformed['BMI'] * X_transformed['Duration']
        X_transformed['Age_BodyTemp'] = X_transformed['Age'] * X_transformed['Body_Temp']
        
        return X_transformed
        
    def get_feature_names_out(self, input_features=None):
        """Return feature names including all new derived features."""
        if input_features is None:
            input_features = config.RAW_NUM_FEATURES + [config.CAT_COL]
        
        # Add all derived features to input features
        output_features = list(input_features) + config.DERIVED_FEATURES
        return np.array(output_features)


def build_preprocessing_pipeline() -> Pipeline:
    """Build the complete preprocessing pipeline."""
    
    # Create the pipeline with proper feature name handling
    pipeline = Pipeline([
        ("winsor", Winsorizer(config.WINSOR_LIMITS)),
        ("feat", FeatureEngineer()),
        ("ct", ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), config.KEEP_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), [config.CAT_COL]),
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
    
    # Ensure directories exist
    config.ensure_directories()
    
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
        # Fallback: construct feature names manually for enhanced features
        num_features = [f"num__{feat}" for feat in config.KEEP_FEATURES]
        cat_features = [f"cat__{config.CAT_COL}_male"]  # Only one category since drop="first"
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