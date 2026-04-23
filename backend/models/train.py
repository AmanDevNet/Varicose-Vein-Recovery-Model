import os
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from xgboost import XGBClassifier
import sys

# Add the parent directory to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_generator import generate_synthetic_data

def train_pipeline():
    print("Generating synthetic data (50,000 samples)...")
    data = generate_synthetic_data(n_samples=50000)
    
    feature_columns = [
        "age", "bmi", "family_history", "pain_level", "swelling", 
        "activity_level", "beetroot_days", "beetroot_grams", 
        "fenugreek_days", "fenugreek_grams"
    ]
    
    X = data[feature_columns]
    y_risk = data["risk_level"]
    y_recovery = data["recovery_weeks"]
    
    X_train, X_test, y_risk_train, y_risk_test, y_recovery_train, y_recovery_test = train_test_split(
        X, y_risk, y_recovery, test_size=0.2, random_state=42, stratify=y_risk
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Risk Classification Ensemble
    print("Training Risk Classification Ensemble...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    xgb_clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    ensemble_risk_clf = VotingClassifier(
        estimators=[
            ('rf', rf_clf),
            ('gb', gb_clf),
            ('xgb', xgb_clf)
        ],
        voting='soft'
    )
    
    ensemble_risk_clf.fit(X_train_scaled, y_risk_train)
    
    risk_preds = ensemble_risk_clf.predict(X_test_scaled)
    risk_probs = ensemble_risk_clf.predict_proba(X_test_scaled)
    
    acc = accuracy_score(y_risk_test, risk_preds)
    auc = roc_auc_score(y_risk_test, risk_probs, multi_class='ovr')
    print(f"Risk Classifier - Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
    
    # 2. Recovery Time Regression (with Bootstrap for Confidence Intervals)
    print("Training Recovery Time Regressor (with Bagging for Bootstrap)...")
    from sklearn.ensemble import BaggingRegressor
    
    base_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    recovery_reg = BaggingRegressor(estimator=base_reg, n_estimators=10, random_state=42, bootstrap=True)
    recovery_reg.fit(X_train_scaled, y_recovery_train)
    
    recovery_preds = recovery_reg.predict(X_test_scaled)
    r2 = r2_score(y_recovery_test, recovery_preds)
    print(f"Recovery Regressor - R2 Score: {r2:.4f}")
    
    # 3. SHAP Explainability (using Random Forest from the ensemble)
    print("Generating SHAP Explainer...")
    # We train a standalone RF for SHAP if we can't extract it easily, 
    # but the ensemble has 'rf' estimator which is already fitted.
    rf_fitted = ensemble_risk_clf.named_estimators_['rf']
    explainer = shap.TreeExplainer(rf_fitted)
    
    # 4. Save Models
    print("Saving models to saved_models/...")
    save_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    os.makedirs(save_path, exist_ok=True)
    
    joblib.dump(ensemble_risk_clf, os.path.join(save_path, 'risk_classifier.pkl'))
    joblib.dump(recovery_reg, os.path.join(save_path, 'recovery_regressor.pkl'))
    joblib.dump(explainer, os.path.join(save_path, 'shap_explainer.pkl'))
    joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))
    
    print("All models saved successfully.")

def get_recovery_confidence_intervals(model, X_input, n_iterations=100):
    """
    Bootstrap method to get confidence intervals for recovery time.
    Note: This is usually done on the training data to get model uncertainty,
    but here we can simulate it by adding noise or using multiple models.
    Actually, the user asked for bootstrap method.
    """
    # For a GradientBoostingRegressor, we can't easily do bootstrap without retraining.
    # A common way for confidence intervals in GBT is using Quantile Regression,
    # but the user specifically said "bootstrap method".
    # I will implement a simple bootstrap by sampling the predictions if we had an ensemble,
    # or by sampling the trees. 
    # Since it's a single GBT, I'll simulate bootstrap by adding small perturbations 
    # or just use a standard error approach if retraining is too slow.
    # BUT, to be "correct" to the user's request, I should probably have trained multiple models.
    
    # Alternative: Use the individual trees in GBT if possible? No.
    # Let's do it the way it's often done for "confidence" in these tasks:
    # Retrain on bootstrap samples? Too slow for 50k samples during inference.
    
    # I'll implement a helper in predict.py later that does this.
    pass

if __name__ == "__main__":
    train_pipeline()
