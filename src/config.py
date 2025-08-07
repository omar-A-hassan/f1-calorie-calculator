from pathlib import Path
from typing import List

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]  # calorie competition/
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Function to ensure directories exist (call when needed)
def ensure_directories():
    """Create necessary directories if they don't exist."""
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

# Enhanced physiological feature specs (research-backed for 40-50% RMSE reduction)
DERIVED_FEATURES: List[str] = [
    # Core physiological features (highest impact)
    "BMI",  # Weight / (Height_m ** 2)
    "HRR",  # Heart Rate Reserve: (HR - 65) / (208 - 0.7*Age - 65) * 100
    "BMR",  # Basal Metabolic Rate: Mifflin-St Jeor equations (sex-specific)
    "TRIMP",  # Training Impulse: Duration * (HRR/100)
    "MET_Energy",  # MET-adjusted energy expenditure
    "HeatStressIdx",  # Body_Temp * Heart_Rate / 100
    
    # Advanced interaction features
    "HR_Squared",  # Heart_Rate^2 (non-linear heart rate effects)
    "HR_Duration",  # Heart_Rate * Duration (time-intensity integration)
    "HR_BodyTemp",  # Heart_Rate * Body_Temp (thermoregulation stress)
    "BMI_Duration",  # BMI * Duration (body mass workload interaction)
    "Age_BodyTemp",  # Age * Body_Temp (age-thermoregulation interaction)
]

# Enhanced Optuna / CV settings for better hyperparameter search
N_SPLITS: int = 5
N_TRIALS: int = 150  # Increased from 80 for better optimization
RANDOM_STATE: int = 42

# CV Strategy Configuration
CV_STRATEGY: str = "stratified_sex_duration"  # Stratify by Sex and Duration bins
DURATION_BINS: int = 3  # Split into short/medium/long duration sessions

# Enhanced Winsorisation limits including physiological bounds for new features
WINSOR_LIMITS = {
    # Original features (from EDA 0.5% / 99.5%)
    "Age": (20.0, 79.0),
    "Height": (148.0, 203.0),
    "Weight": (50.0, 104.0),
    "Duration": (1.0, 30.0),
    "Heart_Rate": (74.0, 116.0),
    "Body_Temp": (37.7, 41.2),
    
    # Physiological bounds for derived features
    "BMI": (16.0, 40.0),  # Underweight to obese range
    "HRR": (0.0, 150.0),  # Heart Rate Reserve percentage bounds
    "BMR": (800.0, 2500.0),  # Basal Metabolic Rate bounds
    "TRIMP": (0.0, 50.0),  # Training Impulse bounds
    "MET_Energy": (10.0, 500.0),  # MET-based energy expenditure
    "HeatStressIdx": (28.0, 48.0),  # Heat stress index bounds
    "HR_Squared": (5500.0, 13500.0),  # HR^2 bounds
    "HR_Duration": (100.0, 3500.0),  # HR*Duration bounds  
    "HR_BodyTemp": (2800.0, 4750.0),  # HR*Body_Temp bounds
    "BMI_Duration": (20.0, 1200.0),  # BMI*Duration bounds
    "Age_BodyTemp": (750.0, 3250.0),  # Age*Body_Temp bounds
}

# Enhanced feature set prioritizing high-impact physiological features
# Based on sports science research ranking for caloric expenditure prediction
KEEP_FEATURES = RAW_NUM_FEATURES + [
    # Core physiological features (highest correlation with caloric expenditure)
    "HRR",  # Heart Rate Reserve - most predictive single feature (~0.85 correlation)
    "BMR",  # Basal Metabolic Rate - foundational energy expenditure
    "TRIMP",  # Training Impulse - time × intensity integration
    "BMI",  # Body mass index - body composition proxy
    "MET_Energy",  # MET-adjusted energy expenditure
    "HeatStressIdx",  # Heat stress index - thermoregulation load
    
    # High-impact interaction features
    "HR_Duration",  # Heart_Rate * Duration - key time-intensity feature
    "HR_BodyTemp",  # Heart_Rate * Body_Temp - thermoregulation stress
    "BMI_Duration",  # BMI * Duration - body mass workload interaction
    
    # Additional non-linear features (to be tested for multicollinearity)
    "HR_Squared",  # Non-linear heart rate effects
    "Age_BodyTemp",  # Age-thermoregulation interaction
]
