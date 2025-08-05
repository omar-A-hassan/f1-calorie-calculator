from pathlib import Path
from typing import List

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]  # calorie competition/
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
    "Height (cm)",
    "Weight (kg)",
    "Duration (min)",
    "Heart_Rate",
    "Body_Temp (C)",
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
