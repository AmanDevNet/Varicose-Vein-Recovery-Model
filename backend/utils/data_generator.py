import numpy as np
import pandas as pd
from typing import Tuple

def generate_synthetic_data(n_samples: int = 50000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic dataset for varicose vein recovery prediction.
    
    Features:
    - age: 20-80
    - bmi: 18-40
    - family_history: 0 or 1
    - pain_level: 0-10
    - swelling_level: 0-10
    - activity_level: 0-10
    - beetroot_days: 0-30
    - beetroot_grams: 0-30
    - fenugreek_days: 0-30
    - fenugreek_grams: 0-20
    
    Targets:
    - risk_level: 0 (Low), 1 (Moderate), 2 (High)
    - recovery_weeks: 2-20 weeks
    """
    np.random.seed(random_state)

    age = np.clip(np.random.normal(50, 15, n_samples), 20, 80)
    bmi = np.clip(np.random.normal(26, 4, n_samples), 18, 40)
    family_history = np.random.binomial(1, 0.3, n_samples)

    pain_level = np.clip(np.random.normal(5 + age / 20 + bmi / 10, 2, n_samples), 0, 10)
    swelling = np.clip(np.random.normal(4 + age / 25 + bmi / 8, 2, n_samples), 0, 10)
    activity_level = np.clip(np.random.normal(6 - age / 30 - bmi / 12, 2, n_samples), 0, 10)

    # Simplified supplement features for the generator, can be mapped to categories later
    beetroot_days = np.clip(np.random.poisson(15, n_samples), 0, 30)
    beetroot_grams = np.clip(np.random.normal(10, 5, n_samples), 0, 30)
    fenugreek_days = np.clip(np.random.poisson(12, n_samples), 0, 30)
    fenugreek_grams = np.clip(np.random.normal(8, 4, n_samples), 0, 20)

    data = pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "family_history": family_history,
        "pain_level": pain_level,
        "swelling": swelling,
        "activity_level": activity_level,
        "beetroot_days": beetroot_days,
        "beetroot_grams": beetroot_grams,
        "fenugreek_days": fenugreek_days,
        "fenugreek_grams": fenugreek_grams,
    })

    # Risk Score Logic
    risk_score = (
        0.3 * (data["age"] - 20) / 60
        + 0.2 * (data["bmi"] - 18) / 22
        + 0.2 * data["family_history"]
        + 0.15 * data["pain_level"] / 10
        + 0.15 * data["swelling"] / 10
        - 0.1 * (data["beetroot_days"] * data["beetroot_grams"]) / 300
        - 0.1 * (data["fenugreek_days"] * data["fenugreek_grams"]) / 200
    )
    risk_level = np.where(risk_score < 0.3, 0, np.where(risk_score < 0.7, 1, 2))

    # Recovery Weeks Logic
    supplement_effect = (
        0.15 * (data["beetroot_days"] * data["beetroot_grams"]) / 100
        + 0.12 * (data["fenugreek_days"] * data["fenugreek_grams"]) / 60
    )
    recovery_weeks = (
        8
        + 0.25 * (data["age"] - 50) / 10
        + 0.18 * (data["bmi"] - 25) / 5
        + 0.12 * data["family_history"] * 4
        + 0.25 * data["pain_level"]
        + 0.22 * data["swelling"]
        - 0.12 * data["activity_level"]
        - supplement_effect
        + 0.05 * (data["pain_level"] * data["swelling"]) / 10
        - 0.05 * (data["activity_level"] * supplement_effect) / 10
        + np.random.normal(0, 0.3, n_samples)
    )
    recovery_weeks = np.clip(recovery_weeks, 2, 20)

    data["risk_level"] = risk_level
    data["recovery_weeks"] = recovery_weeks
    
    return data

if __name__ == "__main__":
    df = generate_synthetic_data(50000)
    print(f"Generated {len(df)} samples.")
    print(df.head())
    print(df.describe())
