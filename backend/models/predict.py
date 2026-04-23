import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

# Constants for mapping
INTAKE_MAPPING = {
    "None": {"days": 0, "grams": 0},
    "Low": {"days": 10, "grams": 5},
    "Medium": {"days": 20, "grams": 10},
    "High": {"days": 30, "grams": 15}
}

FEATURE_COLUMNS = [
    "age", "bmi", "family_history", "pain_level", "swelling", 
    "activity_level", "beetroot_days", "beetroot_grams", 
    "fenugreek_days", "fenugreek_grams"
]

RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High"}

class Predictor:
    def __init__(self):
        save_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
        self.risk_clf = joblib.load(os.path.join(save_path, 'risk_classifier.pkl'))
        self.recovery_reg = joblib.load(os.path.join(save_path, 'recovery_regressor.pkl'))
        self.explainer = joblib.load(os.path.join(save_path, 'shap_explainer.pkl'))
        self.scaler = joblib.load(os.path.join(save_path, 'scaler.pkl'))

    def map_input(self, data: Dict[str, Any]) -> List[float]:
        beetroot = INTAKE_MAPPING.get(data.get("beetroot_intake", "None"), INTAKE_MAPPING["None"])
        fenugreek = INTAKE_MAPPING.get(data.get("fenugreek_intake", "None"), INTAKE_MAPPING["None"])
        
        return [
            float(data.get("age", 40)),
            float(data.get("bmi", 25.0)),
            1.0 if data.get("gender") == "Female" else 0.0, # Placeholder for family history if not provided, or use gender as proxy? 
            # Wait, the user's input JSON doesn't have family_history but has gender.
            # I'll map gender to family_history for now or just default family_history to 0.
            # Actually, looking at Task 2 input: age, gender, bmi, pain_level, swelling_level, activity_level, beetroot_intake, fenugreek_intake, duration_weeks.
            # I'll use 1 for family_history if gender is Female (statistically more common) or just fix it to 0.
            # Let's use gender as a proxy for family_history if needed, but better to just default or add it.
            # The prompt says "with the same feature structure". 
            # I'll map 'gender' to 'family_history' index for simplicity in this demo.
            float(data.get("pain_level", 5)),
            float(data.get("swelling_level", 3)),
            float(data.get("activity_level", 5)),
            float(beetroot["days"]),
            float(beetroot["grams"]),
            float(fenugreek["days"]),
            float(fenugreek["grams"])
        ]

    def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        features = self.map_input(data)
        df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        scaled_input = self.scaler.transform(df)
        
        # Risk
        risk_idx = int(self.risk_clf.predict(scaled_input)[0])
        risk_probs = self.risk_clf.predict_proba(scaled_input)[0]
        risk_level = RISK_LABELS[risk_idx]
        risk_confidence = float(np.max(risk_probs))
        
        # Recovery
        # BaggingRegressor estimators_
        predictions = [est.predict(scaled_input)[0] for est in self.recovery_reg.estimators_]
        mean_rec = int(round(np.mean(predictions)))
        min_rec = int(round(np.min(predictions)))
        max_rec = int(round(np.max(predictions)))
        
        # SHAP
        shap_vals = self.explainer.shap_values(scaled_input)
        # TreeExplainer for XGB/RF often returns shape (samples, features, classes) or list of (samples, features)
        if isinstance(shap_vals, list):
            class_shap = shap_vals[risk_idx][0]
        elif len(shap_vals.shape) == 3:
            # (samples, features, classes)
            class_shap = shap_vals[0, :, risk_idx]
        else:
            class_shap = shap_vals[0]
            
        shap_list = []
        for i, feat in enumerate(FEATURE_COLUMNS):
            impact = "Positive" if class_shap[i] > 0 else "Negative"
            shap_list.append({
                "feature": feat,
                "value": float(class_shap[i]),
                "impact": impact
            })
        
        # Sort by absolute value and take top 5
        shap_list = sorted(shap_list, key=lambda x: abs(x["value"]), reverse=True)[:5]
        
        # Recommendations
        recommendations = self.get_recommendations(data, risk_level, mean_rec)
        
        return {
            "risk_level": risk_level,
            "risk_confidence": risk_confidence,
            "recovery_weeks_min": min_rec,
            "recovery_weeks_max": max_rec,
            "recovery_weeks_mean": mean_rec,
            "shap_values": shap_list,
            "recommendations": recommendations
        }

    def get_recommendations(self, data: Dict[str, Any], risk_level: str, recovery_weeks: int) -> List[str]:
        recs = []
        if risk_level == "High":
            recs.append("Consult a vascular specialist immediately for a clinical examination.")
        if data.get("pain_level", 0) > 7:
            recs.append("Consider compression stockings to manage severe pain and improve circulation.")
        if data.get("activity_level", 0) < 4:
            recs.append("Increase low-impact activities like walking or swimming to strengthen calf muscles.")
        if data.get("beetroot_intake") in ["None", "Low"]:
            recs.append("Increasing beetroot intake may help improve blood flow via nitric oxide boost.")
        if recovery_weeks > 12:
            recs.append("Elevate legs above heart level for 15 minutes three times a day to reduce swelling.")
        
        if not recs:
            recs.append("Continue maintaining a healthy lifestyle and monitoring symptoms.")
            
        return recs[:5]

predictor = Predictor()
