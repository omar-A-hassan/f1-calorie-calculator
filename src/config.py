from pathlib import Path
from typing import List

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]  # calorie competition/
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Ensure directories exist
for _p in [DATA_PROCESSED, MODELS_DIR, REPORTS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# Core feature list (raw columns expected in CSV)
RAW_NUM_FEATURES: List[str] = [
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
]
TARGET_COL: str = "Calories"
CAT_COL: str = "Sex"  # 'male' / 'female'

# Derived feature specs
DERIVED_FEATURES: List[str] = [
    "BMI",  # Weight / (Height_m ** 2)
    "HRxDur",  # Heart_Rate * Duration
    "Age_x_BodyTemp",  # Age * Body_Temp
    "HeatStressIdx",  # Body_Temp * Heart_Rate / 100
]

# Optuna / CV settings
N_SPLITS: int = 5
N_TRIALS: int = 80  # per model
RANDOM_STATE: int = 42

# ---- Winsorisation cut-offs discovered in EDA (0.5 % / 99.5 %)
WINSOR_LIMITS = {
    "Age": (20.0, 79.0),
    "Height": (148.0, 203.0),
    "Weight": (50.0, 104.0),
    "Duration": (1.0, 30.0),
    "Heart_Rate": (74.0, 116.0),
    "Body_Temp": (37.7, 41.2),
}

# Final feature set to keep after multicollinearity pruning
KEEP_FEATURES = RAW_NUM_FEATURES + [
    "BMI",
    "HeatStressIdx",
    # HRxDur and Age_x_BodyTemp were dropped due to VIF / high ρ
]
