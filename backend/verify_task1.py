import os
import joblib
import numpy as np
import pandas as pd

def verify_models():
    save_path = 'backend/saved_models'
    
    print("Loading models...")
    risk_clf = joblib.load(os.path.join(save_path, 'risk_classifier.pkl'))
    recovery_reg = joblib.load(os.path.join(save_path, 'recovery_regressor.pkl'))
    explainer = joblib.load(os.path.join(save_path, 'shap_explainer.pkl'))
    scaler = joblib.load(os.path.join(save_path, 'scaler.pkl'))
    
    # Sample input
    sample_data = {
        "age": 45, "bmi": 24.5, "family_history": 1, 
        "pain_level": 5, "swelling": 3, "activity_level": 7, 
        "beetroot_days": 15, "beetroot_grams": 10, 
        "fenugreek_days": 10, "fenugreek_grams": 5
    }
    df = pd.DataFrame([sample_data])
    scaled_input = scaler.transform(df)
    
    # 1. Risk Prediction
    risk_pred = risk_clf.predict(scaled_input)[0]
    risk_proba = risk_clf.predict_proba(scaled_input)[0]
    print(f"Risk Prediction: {risk_pred} (Proba: {risk_proba})")
    
    # 2. Recovery Prediction with Bootstrap (Confidence Intervals)
    # The recovery_reg is a BaggingRegressor
    predictions = [est.predict(scaled_input)[0] for est in recovery_reg.estimators_]
    mean_rec = np.mean(predictions)
    min_rec = np.min(predictions)
    max_rec = np.max(predictions)
    print(f"Recovery (weeks): Mean={mean_rec:.2f}, Min={min_rec:.2f}, Max={max_rec:.2f}")
    
    # 3. SHAP Values
    shap_values = explainer.shap_values(scaled_input)
    # SHAP returns a list for multiclass
    print(f"SHAP Values Shape: {np.array(shap_values).shape}")
    
    print("Verification complete!")

if __name__ == "__main__":
    verify_models()
