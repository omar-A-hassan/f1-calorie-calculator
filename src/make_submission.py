"""Predict test set with stacked model and create submission.csv."""
import pandas as pd
import joblib
import sys
from pathlib import Path

# Add project root to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

import src.config as cfg


def main() -> None:
    # Ensure data directories exist
    cfg.ensure_directories()
    
    try:
        # Load processed test features
        X_test = pd.read_parquet(cfg.DATA_PROCESSED / "test.parquet")
        print(f"Loaded processed test features: {X_test.shape}")
        
    except FileNotFoundError as e:
        print(f"ERROR: Required test data file not found: {e}")
        exit(1)
    
    # Load base models and generate meta-features
    base_model_names = ["lgb", "cat", "xgb", "enet"]
    meta_features = {}
    
    print("Loading base models and generating meta-features:")
    for model_name in base_model_names:
        model_path = cfg.MODELS_DIR / f"{model_name}.pkl"
        try:
            model = joblib.load(model_path)
            pred = model.predict(X_test)
            meta_features[f"pred_{model_name}"] = pred
            print(f"  ✓ {model_name}: generated {len(pred)} predictions")
        except FileNotFoundError:
            print(f"  ⚠️  {model_name}: model file not found, skipping")
            continue
        except Exception as e:
            print(f"  ✗ {model_name}: error loading/predicting - {e}")
            continue
    
    if not meta_features:
        print("ERROR: No base models could be loaded")
        exit(1)
    
    # Build meta-features DataFrame
    meta_test = pd.DataFrame(meta_features)
    print(f"Created meta-features DataFrame: {meta_test.shape}")
    
    try:
        # Load the stacked model
        stack_model = joblib.load(cfg.MODELS_DIR / "stacked.pkl")
        print("✓ Loaded stacked model")
        
        # Try to preserve column order from training
        if hasattr(stack_model, 'feature_names_in_'):
            desired_order = [col for col in stack_model.feature_names_in_ if col in meta_test.columns]
            if desired_order:
                meta_test = meta_test[desired_order]
                print(f"  Using training column order: {desired_order}")
            else:
                meta_test = meta_test.sort_index(axis=1)
                print(f"  Using alphabetical order: {list(meta_test.columns)}")
        else:
            meta_test = meta_test.sort_index(axis=1)
            print(f"  Using alphabetical order: {list(meta_test.columns)}")
            
    except FileNotFoundError:
        print("ERROR: Stacked model not found at models/stacked.pkl")
        print("Please run stacking.py first to train the meta-learner")
        exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load stacked model - {e}")
        exit(1)
    
    # Generate final predictions using meta-features
    try:
        test_pred = stack_model.predict(meta_test)
        print(f"Generated final predictions: {len(test_pred)} samples")
    except Exception as e:
        print(f"ERROR: Failed to generate predictions - {e}")
        exit(1)
    
    try:
        # Read sample submission to preserve id order
        sample_sub = pd.read_csv(cfg.DATA_RAW / "sample_submission.csv")
        print(f"Loaded sample submission template: {sample_sub.shape}")
        
        # Create submission with same id order as sample
        submission = sample_sub.copy()
        submission["Calories"] = test_pred
        
        # Verify alignment
        if len(submission) != len(test_pred):
            print(f"ERROR: Length mismatch - submission: {len(submission)}, predictions: {len(test_pred)}")
            exit(1)
            
    except FileNotFoundError:
        print("ERROR: sample_submission.csv not found")
        exit(1)
    except Exception as e:
        print(f"ERROR: Failed to process sample submission - {e}")
        exit(1)
    
    # Save submission to project root
    output_path = cfg.PROJECT_ROOT / "submission.csv"
    try:
        submission.to_csv(output_path, index=False)
        print(f"[make_submission] Submission saved to {output_path}, shape={submission.shape}")
    except Exception as e:
        print(f"ERROR: Failed to save submission - {e}")
        exit(1)


if __name__ == "__main__":
    main()