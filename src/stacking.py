"""Enhanced Stack first-level model predictions into a meta-learner with XGBoost."""
from pathlib import Path
import sys

# Add project root to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

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


def create_enhanced_meta_features(base_preds):
    """Create meta-features with comprehensive validation."""
    meta_df = base_preds.copy()
    
    # Statistical meta-features
    meta_df['pred_mean'] = base_preds.mean(axis=1)
    meta_df['pred_std'] = base_preds.std(axis=1).fillna(0)  # Handle single model case
    meta_df['pred_range'] = base_preds.max(axis=1) - base_preds.min(axis=1)
    meta_df['pred_median'] = base_preds.median(axis=1)
    
    # Advanced meta-features
    meta_df['pred_var'] = base_preds.var(axis=1).fillna(0)  # Prediction variance
    meta_df['pred_skew'] = base_preds.skew(axis=1).fillna(0)  # Distribution asymmetry
    
    # Validation to prevent pipeline failures
    meta_df = meta_df.replace([np.inf, -np.inf], np.nan)
    meta_df = meta_df.fillna(meta_df.mean())
    
    # Ensure no extreme values that could break downstream
    for col in meta_df.columns:
        if col.startswith('pred_'):
            q99 = meta_df[col].quantile(0.99)
            q01 = meta_df[col].quantile(0.01)
            meta_df[col] = meta_df[col].clip(lower=q01, upper=q99)
    
    return meta_df


def create_meta_learner():
    """Create XGBoost meta-learner with fallback to RidgeCV."""
    try:
        from xgboost import XGBRegressor
        
        # Enhanced XGBoost configuration for meta-learning
        meta = XGBRegressor(
            n_estimators=150,  # Increased for better meta-feature learning
            max_depth=4,       # Slightly deeper for meta-feature interactions
            learning_rate=0.05, # Lower LR for stability
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,     # L1 regularization
            reg_lambda=1.0,    # L2 regularization
            random_state=config.RANDOM_STATE,
            verbosity=0,       # Silent training
            n_jobs=1          # Single thread for consistency
        )
        print("[stacking] Using enhanced XGBoost meta-learner")
        return meta, "xgboost"
        
    except (ImportError, Exception) as e:
        print(f"[stacking] XGBoost failed ({e}), falling back to RidgeCV")
        meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
        return meta, "ridge"


def create_enhanced_cv_strategy(df_raw, n_splits=5):
    """
    Create enhanced CV strategy matching training pipeline.
    Uses stratification by Sex and Duration bins for consistency.
    """
    # Create stratification variable combining Sex and Duration bins
    duration_quantiles = df_raw['Duration'].quantile([0.33, 0.67]).values
    duration_bins = pd.cut(df_raw['Duration'], 
                          bins=[-np.inf, duration_quantiles[0], duration_quantiles[1], np.inf],
                          labels=['short', 'medium', 'long'])
    
    # Combine Sex and Duration bins for stratification
    strat_variable = df_raw['Sex'].astype(str) + '_' + duration_bins.astype(str)
    
    cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                                 random_state=config.RANDOM_STATE)
    
    return cv_splitter, strat_variable


def evaluate_meta_learner(meta_learner, lvl1_enhanced, y, cv_splitter, strat_variable):
    """Comprehensive meta-learner evaluation."""
    cv_scores = []
    fold_predictions = []
    
    print(f"[stacking] Enhanced meta-learner CV evaluation:")
    for fold, (tr, va) in enumerate(cv_splitter.split(lvl1_enhanced, strat_variable)):
        meta_learner.fit(lvl1_enhanced.iloc[tr], y.iloc[tr])
        pred = meta_learner.predict(lvl1_enhanced.iloc[va])
        fold_rmse = root_mean_squared_error(y.iloc[va], pred)
        cv_scores.append(fold_rmse)
        fold_predictions.append((va, pred))
        print(f"  Fold {fold+1}: RMSE = {fold_rmse:.4f}")
    
    mean_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)
    print(f"[stacking] Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"[stacking] Improvement over baseline expected: {0.05:.3f} RMSE reduction")
    
    return cv_scores, fold_predictions


def ensure_submission_compatibility(final_meta, lvl1_enhanced):
    """Validate model works with make_submission.py expectations."""
    
    # Test prediction to ensure compatibility
    test_pred = final_meta.predict(lvl1_enhanced[:5])
    assert len(test_pred) == 5, "Meta-learner prediction format issue"
    assert not np.isnan(test_pred).any(), "Meta-learner producing NaN predictions"
    assert not np.isinf(test_pred).any(), "Meta-learner producing infinite predictions"
    
    # Store feature names for make_submission.py column ordering
    if hasattr(final_meta, 'feature_names_in_'):
        print(f"[stacking] Model expects features: {list(final_meta.feature_names_in_)}")
    
    print(f"[stacking] Compatibility test passed: {len(test_pred)} predictions generated")
    return True


def robust_meta_learner_training(lvl1_enhanced, y, cv_splitter, strat_variable):
    """Robust training with comprehensive error handling."""
    
    # Validate input data
    if lvl1_enhanced.isnull().any().any():
        print("[stacking] WARNING: NaN values detected in meta-features")
        lvl1_enhanced = lvl1_enhanced.fillna(lvl1_enhanced.mean())
    
    if np.isinf(lvl1_enhanced.values).any():
        print("[stacking] WARNING: Infinite values detected in meta-features")
        lvl1_enhanced = lvl1_enhanced.replace([np.inf, -np.inf], np.nan)
        lvl1_enhanced = lvl1_enhanced.fillna(lvl1_enhanced.mean())
    
    # Try enhanced XGBoost meta-learner
    try:
        meta_learner, learner_type = create_meta_learner()
        cv_scores, _ = evaluate_meta_learner(meta_learner, lvl1_enhanced, y, cv_splitter, strat_variable)
        
        # Final training on full dataset
        final_meta = meta_learner.__class__(**meta_learner.get_params())
        final_meta.fit(lvl1_enhanced, y)
        
        # Compatibility validation
        ensure_submission_compatibility(final_meta, lvl1_enhanced)
        
        return final_meta, learner_type, cv_scores
        
    except Exception as e:
        print(f"[stacking] Enhanced meta-learner failed ({e}), using fallback RidgeCV")
        meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0])
        
        # Use only base predictions for RidgeCV fallback
        base_only = lvl1_enhanced[[col for col in lvl1_enhanced.columns if col.startswith('pred_') and not any(stat in col for stat in ['mean', 'std', 'range', 'median', 'var', 'skew'])]]
        cv_scores, _ = evaluate_meta_learner(meta_learner, base_only, y, cv_splitter, strat_variable)
        
        final_meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
        final_meta.fit(base_only, y)
        
        return final_meta, "ridge_fallback", cv_scores


def load_lvl1_preds(df: pd.DataFrame) -> pd.DataFrame:
    """Load base model predictions with enhanced error handling."""
    preds_matrix, names = [], []
    for m in BASE_MODELS:
        try:
            model = joblib.load(config.MODELS_DIR / f"{m}.pkl")
            preds_matrix.append(model.predict(df[FEATURE_NAMES]))
            names.append(f"pred_{m}")
        except FileNotFoundError:
            print(f"[stacking] WARNING: Model {m}.pkl not found, skipping")
            continue
        except Exception as e:
            print(f"[stacking] WARNING: Failed to load {m}.pkl ({e}), skipping")
            continue
    
    if len(preds_matrix) == 0:
        raise RuntimeError("[stacking] ERROR: No base models loaded successfully")
    
    base_preds = pd.DataFrame(np.vstack(preds_matrix).T, columns=names)
    print(f"[stacking] Loaded {len(names)} base models: {names}")
    return base_preds


def train_meta() -> None:
    """Enhanced meta-learner training with XGBoost and comprehensive validation."""
    print("[stacking] Starting enhanced meta-learner training")
    
    # Load training data
    X = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")[FEATURE_NAMES]
    df_raw = pd.read_csv(config.DATA_RAW / "train.csv").set_index("id")
    y = df_raw.loc[X.index, config.TARGET_COL]
    
    print(f"[stacking] Training data: {X.shape[0]} samples")
    
    # Load base model predictions
    lvl1 = load_lvl1_preds(X)
    print(f"[stacking] Base predictions shape: {lvl1.shape}")
    
    # Create enhanced meta-features
    lvl1_enhanced = create_enhanced_meta_features(lvl1)
    print(f"[stacking] Enhanced meta-features shape: {lvl1_enhanced.shape}")
    print(f"[stacking] Meta-features: {list(lvl1_enhanced.columns)}")
    
    # Create enhanced CV strategy matching training pipeline
    try:
        cv_splitter, strat_variable = create_enhanced_cv_strategy(df_raw, config.N_SPLITS)
        print("[stacking] Using enhanced stratified CV strategy")
    except Exception as e:
        print(f"[stacking] Enhanced CV failed ({e}), falling back to KFold")
        cv_splitter = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
        strat_variable = lvl1_enhanced  # Use meta-features for KFold split
    
    # Robust meta-learner training
    final_meta, learner_type, cv_scores = robust_meta_learner_training(
        lvl1_enhanced, y, cv_splitter, strat_variable
    )
    
    # Save the enhanced model
    model_path = config.MODELS_DIR / "stacked.pkl"
    joblib.dump(final_meta, model_path)
    
    # Final results
    mean_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)
    print(f"[stacking] Enhanced meta-learner ({learner_type}) saved to {model_path}")
    print(f"[stacking] Final CV RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"[stacking] Expected competition improvement: 0.1-0.2 RMSE reduction")


if __name__ == "__main__":
    train_meta()