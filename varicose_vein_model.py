import math
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

try:
    import requests
except Exception:
    requests = None

warnings.filterwarnings("ignore")


class VaricoseVeinMLModel:
    """Machine learning model for varicose vein risk and recovery estimation."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.risk_classifier: Optional[RandomForestClassifier] = None
        self.recovery_regressor: Optional[GradientBoostingRegressor] = None
        self.scaler = StandardScaler()
        self.metrics_: Dict[str, Any] = {}

        self.feature_columns = [
            "age",
            "bmi",
            "family_history",
            "pain_level",
            "swelling",
            "activity_level",
            "beetroot_days_30",
            "beetroot_grams_day",
            "fenugreek_days_30",
            "fenugreek_grams_day",
        ]
        self.input_ranges: Dict[str, Tuple[float, float]] = {
            "age": (20.0, 80.0),
            "bmi": (15.0, 45.0),
            "family_history": (0.0, 1.0),
            "pain_level": (0.0, 10.0),
            "swelling": (0.0, 10.0),
            "activity_level": (0.0, 10.0),
            "beetroot_days_30": (0.0, 30.0),
            "beetroot_grams_day": (0.0, 30.0),
            "fenugreek_days_30": (0.0, 30.0),
            "fenugreek_grams_day": (0.0, 20.0),
        }
        self.risk_labels = {0: "Low", 1: "Moderate", 2: "High"}

    def generate_synthetic_data(self, n_samples: int = 8000) -> pd.DataFrame:
        """Generate synthetic dataset for model development."""
        np.random.seed(self.random_state)

        age = np.clip(np.random.normal(50, 15, n_samples), 20, 80)
        bmi = np.clip(np.random.normal(26, 4, n_samples), 18, 40)
        family_history = np.random.binomial(1, 0.3, n_samples)

        pain_level = np.clip(np.random.normal(5 + age / 20 + bmi / 10, 2, n_samples), 0, 10)
        swelling = np.clip(np.random.normal(4 + age / 25 + bmi / 8, 2, n_samples), 0, 10)
        activity_level = np.clip(np.random.normal(6 - age / 30 - bmi / 12, 2, n_samples), 0, 10)

        beetroot_days_30 = np.clip(np.random.poisson(15, n_samples), 0, 30)
        beetroot_grams_day = np.clip(np.random.normal(10, 5, n_samples), 0, 30)
        fenugreek_days_30 = np.clip(np.random.poisson(12, n_samples), 0, 30)
        fenugreek_grams_day = np.clip(np.random.normal(8, 4, n_samples), 0, 20)

        data = pd.DataFrame(
            {
                "age": age,
                "bmi": bmi,
                "family_history": family_history,
                "pain_level": pain_level,
                "swelling": swelling,
                "activity_level": activity_level,
                "beetroot_days_30": beetroot_days_30,
                "beetroot_grams_day": beetroot_grams_day,
                "fenugreek_days_30": fenugreek_days_30,
                "fenugreek_grams_day": fenugreek_grams_day,
            }
        )

        risk_score = (
            0.3 * (data["age"] - 20) / 60
            + 0.2 * (data["bmi"] - 18) / 22
            + 0.2 * data["family_history"]
            + 0.15 * data["pain_level"] / 10
            + 0.15 * data["swelling"] / 10
            - 0.1 * (data["beetroot_days_30"] * data["beetroot_grams_day"]) / 300
            - 0.1 * (data["fenugreek_days_30"] * data["fenugreek_grams_day"]) / 200
        )
        risk_level = np.where(risk_score < 0.3, 0, np.where(risk_score < 0.7, 1, 2))

        supplement_effect = (
            0.15 * (data["beetroot_days_30"] * data["beetroot_grams_day"]) / 100
            + 0.12 * (data["fenugreek_days_30"] * data["fenugreek_grams_day"]) / 60
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

    def artifacts_exist(self, directory: str = ".") -> bool:
        required = ["risk_classifier.pkl", "recovery_regressor.pkl", "scaler.pkl"]
        return all(os.path.exists(os.path.join(directory, name)) for name in required)

    def save_artifacts(self, directory: str = ".") -> None:
        if self.risk_classifier is None or self.recovery_regressor is None:
            raise RuntimeError("Models are not trained. Train or load models before saving.")
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.risk_classifier, os.path.join(directory, "risk_classifier.pkl"))
        joblib.dump(self.recovery_regressor, os.path.join(directory, "recovery_regressor.pkl"))
        joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))

    def load_artifacts(self, directory: str = ".") -> None:
        self.risk_classifier = joblib.load(os.path.join(directory, "risk_classifier.pkl"))
        self.recovery_regressor = joblib.load(os.path.join(directory, "recovery_regressor.pkl"))
        self.scaler = joblib.load(os.path.join(directory, "scaler.pkl"))

    def _train_risk_classifier(
        self,
        X_train_scaled: np.ndarray,
        y_risk_train: pd.Series,
        enable_tuning: bool,
    ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        if not enable_tuning:
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_train_scaled, y_risk_train)
            return model, {}

        param_dist = {
            "n_estimators": [150, 250, 300, 400, 500],
            "max_depth": [8, 12, 15, 20, None],
            "min_samples_split": [2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3, 4],
            "max_features": ["sqrt", "log2", None],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            param_distributions=param_dist,
            n_iter=20,
            scoring="f1_weighted",
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
        )
        search.fit(X_train_scaled, y_risk_train)
        return search.best_estimator_, search.best_params_

    def _train_recovery_regressor(
        self,
        X_train_scaled: np.ndarray,
        y_recovery_train: pd.Series,
        enable_tuning: bool,
    ) -> Tuple[GradientBoostingRegressor, Dict[str, Any]]:
        if not enable_tuning:
            model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=self.random_state,
            )
            model.fit(X_train_scaled, y_recovery_train)
            return model, {}

        param_dist = {
            "n_estimators": [150, 250, 300, 400, 500],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "max_depth": [2, 3, 4, 5, 6],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "min_samples_split": [2, 4, 6, 8],
            "min_samples_leaf": [1, 2, 3],
        }
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            estimator=GradientBoostingRegressor(random_state=self.random_state),
            param_distributions=param_dist,
            n_iter=20,
            scoring="r2",
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
        )
        search.fit(X_train_scaled, y_recovery_train)
        return search.best_estimator_, search.best_params_

    def train_models(
        self,
        data: pd.DataFrame,
        enable_tuning: bool = True,
        save_artifacts: bool = True,
        verbose: bool = True,
        compute_cv: bool = True,
    ) -> Tuple[np.ndarray, pd.Series, pd.Series, np.ndarray, np.ndarray]:
        """Train classification and regression models and store evaluation metrics."""
        X = data[self.feature_columns]
        y_risk = data["risk_level"]
        y_recovery = data["recovery_weeks"]

        X_train, X_test, y_risk_train, y_risk_test, y_recovery_train, y_recovery_test = train_test_split(
            X,
            y_risk,
            y_recovery,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_risk,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.risk_classifier, risk_best_params = self._train_risk_classifier(
            X_train_scaled,
            y_risk_train,
            enable_tuning=enable_tuning,
        )
        self.recovery_regressor, reg_best_params = self._train_recovery_regressor(
            X_train_scaled,
            y_recovery_train,
            enable_tuning=enable_tuning,
        )

        risk_pred = self.risk_classifier.predict(X_test_scaled)
        risk_proba = self.risk_classifier.predict_proba(X_test_scaled)
        recovery_pred = self.recovery_regressor.predict(X_test_scaled)

        accuracy = accuracy_score(y_risk_test, risk_pred)
        precision = precision_score(y_risk_test, risk_pred, average="weighted", zero_division=0)
        recall = recall_score(y_risk_test, risk_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_risk_test, risk_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_risk_test, risk_pred, average="macro", zero_division=0)

        try:
            auc_ovr = roc_auc_score(
                y_risk_test,
                risk_proba,
                multi_class="ovr",
                average="weighted",
            )
        except ValueError:
            auc_ovr = float("nan")

        r2 = r2_score(y_recovery_test, recovery_pred)
        mse = mean_squared_error(y_recovery_test, recovery_pred)

        cv_accuracy_mean: Optional[float] = None
        cv_accuracy_std: Optional[float] = None
        cv_f1_mean: Optional[float] = None
        cv_f1_std: Optional[float] = None
        cv_r2_mean: Optional[float] = None
        cv_r2_std: Optional[float] = None

        if compute_cv:
            clf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            reg_cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

            cv_accuracy = cross_val_score(
                self.risk_classifier,
                X_train_scaled,
                y_risk_train,
                cv=clf_cv,
                scoring="accuracy",
                n_jobs=-1,
            )
            cv_f1 = cross_val_score(
                self.risk_classifier,
                X_train_scaled,
                y_risk_train,
                cv=clf_cv,
                scoring="f1_weighted",
                n_jobs=-1,
            )
            cv_r2 = cross_val_score(
                self.recovery_regressor,
                X_train_scaled,
                y_recovery_train,
                cv=reg_cv,
                scoring="r2",
                n_jobs=-1,
            )
            cv_accuracy_mean = float(cv_accuracy.mean())
            cv_accuracy_std = float(cv_accuracy.std())
            cv_f1_mean = float(cv_f1.mean())
            cv_f1_std = float(cv_f1.std())
            cv_r2_mean = float(cv_r2.mean())
            cv_r2_std = float(cv_r2.std())

        report_text = classification_report(y_risk_test, risk_pred, digits=3, zero_division=0)

        self.metrics_ = {
            "classification": {
                "accuracy": float(accuracy),
                "precision_weighted": float(precision),
                "recall_weighted": float(recall),
                "f1_weighted": float(f1_weighted),
                "f1_macro": float(f1_macro),
                "auc_roc_ovr_weighted": float(auc_ovr) if not np.isnan(auc_ovr) else None,
                "cv_accuracy_mean": cv_accuracy_mean,
                "cv_accuracy_std": cv_accuracy_std,
                "cv_f1_mean": cv_f1_mean,
                "cv_f1_std": cv_f1_std,
                "report": report_text,
            },
            "regression": {
                "r2": float(r2),
                "mse": float(mse),
                "cv_r2_mean": cv_r2_mean,
                "cv_r2_std": cv_r2_std,
            },
            "best_params": {
                "risk_classifier": risk_best_params,
                "recovery_regressor": reg_best_params,
            },
        }

        if verbose:
            print("=== MODEL PERFORMANCE ===")
            if risk_best_params:
                print(f"Best classifier params: {risk_best_params}")
            if reg_best_params:
                print(f"Best regressor params: {reg_best_params}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision (weighted): {precision:.3f}")
            print(f"Recall (weighted): {recall:.3f}")
            print(f"F1 score (weighted): {f1_weighted:.3f}")
            print(f"F1 score (macro): {f1_macro:.3f}")
            if not np.isnan(auc_ovr):
                print(f"AUC-ROC (OvR weighted): {auc_ovr:.3f}")
            print(f"Recovery R2: {r2:.3f}")
            print(f"Recovery MSE: {mse:.3f}")
            if compute_cv and cv_accuracy_mean is not None and cv_accuracy_std is not None:
                print(f"CV accuracy (5-fold): {cv_accuracy_mean:.3f} +/- {cv_accuracy_std:.3f}")
            if compute_cv and cv_f1_mean is not None and cv_f1_std is not None:
                print(f"CV F1 (5-fold): {cv_f1_mean:.3f} +/- {cv_f1_std:.3f}")
            if compute_cv and cv_r2_mean is not None and cv_r2_std is not None:
                print(f"Recovery CV R2 (5-fold): {cv_r2_mean:.3f} +/- {cv_r2_std:.3f}")
            print("\nClassification report:")
            print(report_text)

        if save_artifacts:
            self.save_artifacts()

        return X_test_scaled, y_risk_test, y_recovery_test, risk_pred, recovery_pred

    def _ensure_models_ready(self) -> None:
        if self.risk_classifier is None or self.recovery_regressor is None:
            raise RuntimeError("Models are not available. Train models or load artifacts first.")

    def _sanitize_user_input(self, user_input: Sequence[float]) -> List[float]:
        if len(user_input) != len(self.feature_columns):
            raise ValueError(f"Expected {len(self.feature_columns)} features, got {len(user_input)}.")

        cleaned: List[float] = []
        for idx, col in enumerate(self.feature_columns):
            value = float(user_input[idx])
            low, high = self.input_ranges[col]
            value = max(low, min(high, value))
            if col == "family_history":
                value = 1.0 if value >= 0.5 else 0.0
            cleaned.append(value)
        return cleaned

    def predict_single_case(
        self,
        user_input: Sequence[float],
        return_probabilities: bool = False,
    ):
        """Predict risk level and recovery weeks for one patient profile."""
        self._ensure_models_ready()
        cleaned_input = self._sanitize_user_input(user_input)
        input_df = pd.DataFrame([cleaned_input], columns=self.feature_columns)
        input_scaled = self.scaler.transform(input_df)

        risk_idx = int(self.risk_classifier.predict(input_scaled)[0])
        recovery_pred = float(self.recovery_regressor.predict(input_scaled)[0])
        risk_label = self.risk_labels[risk_idx]

        proba_raw = self.risk_classifier.predict_proba(input_scaled)[0]
        proba_map = {
            self.risk_labels[int(cls_idx)]: float(prob)
            for cls_idx, prob in zip(self.risk_classifier.classes_, proba_raw)
        }
        for label in self.risk_labels.values():
            proba_map.setdefault(label, 0.0)

        if return_probabilities:
            return risk_label, recovery_pred, proba_map
        return risk_label, recovery_pred

    def generate_suggestions(
        self,
        user_input: Sequence[float],
        risk_level: str,
        recovery_weeks: float,
    ) -> List[str]:
        """Generate structured lifestyle suggestions."""
        x = self._sanitize_user_input(user_input)
        suggestions: List[str] = []

        if x[0] > 60:
            suggestions.append("Prioritize low-impact activity such as walking, calf raises, and mobility drills.")
        if x[1] > 25:
            suggestions.append("Weight management may reduce venous pressure and improve symptom control.")
        if x[3] > 6 or x[4] > 6:
            suggestions.append("Use leg elevation and compression support to reduce pain and swelling.")
        if x[5] < 5:
            suggestions.append("Increase daily movement gradually to support circulation.")
        if x[6] < 20 or x[7] < 8:
            suggestions.append("Improve beetroot consistency and dose within safe dietary limits.")
        if x[8] < 15 or x[9] < 6:
            suggestions.append("Improve fenugreek consistency and dose within safe dietary limits.")
        if risk_level == "High":
            suggestions.append("Consult a vascular specialist for detailed diagnostic workup.")
        if recovery_weeks > 12:
            suggestions.append("Discuss long-recovery risk factors with a clinician for a tailored care plan.")
        if not suggestions:
            suggestions.append("Maintain consistency in activity, symptom monitoring, and supplement adherence.")

        return suggestions[:6]

    def _evaluate_regimen(self, base_input: Sequence[float], regimen: Sequence[float]) -> Dict[str, Any]:
        test_input = list(self._sanitize_user_input(base_input))
        test_input[6:10] = regimen
        risk, recovery = self.predict_single_case(test_input)
        return {
            "risk": risk,
            "recovery_weeks": float(recovery),
            "regimen": {
                "beetroot_days_30": int(regimen[0]),
                "beetroot_grams_day": float(regimen[1]),
                "fenugreek_days_30": int(regimen[2]),
                "fenugreek_grams_day": float(regimen[3]),
            },
        }

    def _find_best_regimen(self, base_input: Sequence[float]) -> Dict[str, Any]:
        self._ensure_models_ready()
        base_clean = self._sanitize_user_input(base_input)

        beetroot_days_grid = [0, 10, 15, 20, 25, 30]
        beetroot_grams_grid = [0, 8, 10, 12, 15, 20]
        fenugreek_days_grid = [0, 10, 15, 20, 25, 30]
        fenugreek_grams_grid = [0, 6, 8, 10, 12, 15, 20]

        regimen_grid: List[List[float]] = []
        for bd in beetroot_days_grid:
            for bg in beetroot_grams_grid:
                for fd in fenugreek_days_grid:
                    for fg in fenugreek_grams_grid:
                        regimen_grid.append([float(bd), float(bg), float(fd), float(fg)])

        batch = np.tile(np.array(base_clean, dtype=float), (len(regimen_grid), 1))
        regimen_array = np.array(regimen_grid, dtype=float)
        batch[:, 6:10] = regimen_array

        batch_df = pd.DataFrame(batch, columns=self.feature_columns)
        batch_scaled = self.scaler.transform(batch_df)

        risk_idx = self.risk_classifier.predict(batch_scaled).astype(int)
        risk_names = np.array([self.risk_labels[idx] for idx in risk_idx])
        recovery_pred = self.recovery_regressor.predict(batch_scaled)

        risk_rank = np.array([0 if name == "Low" else 1 if name == "Moderate" else 2 for name in risk_names], dtype=float)
        score = recovery_pred + 0.5 * risk_rank
        best_idx = int(np.argmin(score))

        best_regimen = regimen_array[best_idx]
        return {
            "risk": str(risk_names[best_idx]),
            "recovery_weeks": float(recovery_pred[best_idx]),
            "regimen": {
                "beetroot_days_30": int(best_regimen[0]),
                "beetroot_grams_day": float(best_regimen[1]),
                "fenugreek_days_30": int(best_regimen[2]),
                "fenugreek_grams_day": float(best_regimen[3]),
            },
        }

    def compare_supplement_effect(self, user_input: Sequence[float]) -> Dict[str, Dict[str, Any]]:
        """Compare current regimen, no supplements, and optimized regimen."""
        current = self._evaluate_regimen(
            user_input,
            [
                float(user_input[6]),
                float(user_input[7]),
                float(user_input[8]),
                float(user_input[9]),
            ],
        )
        no_supplements = self._evaluate_regimen(user_input, [0, 0, 0, 0])
        optimal_supplements = self._find_best_regimen(user_input)

        return {
            "current": current,
            "no_supplements": no_supplements,
            "optimal_supplements": optimal_supplements,
        }

    def build_recovery_trajectory(self, user_input: Sequence[float], recovery_weeks: float) -> pd.DataFrame:
        """Build weekly trend data for pain, swelling, and activity."""
        x = self._sanitize_user_input(user_input)
        weeks = np.arange(0, int(math.ceil(max(2.0, recovery_weeks))) + 1)

        supplement_score = min(
            1.0,
            ((x[6] * x[7]) / 450.0) + ((x[8] * x[9]) / 360.0),
        )
        speed_multiplier = 1.0 + 0.35 * supplement_score

        pain = np.clip(x[3] * np.exp(-0.14 * speed_multiplier * weeks), 0, 10)
        swelling = np.clip(x[4] * np.exp(-0.11 * speed_multiplier * weeks), 0, 10)
        activity = np.clip(
            x[5] + (10 - x[5]) * (1 - np.exp(-0.09 * speed_multiplier * weeks)),
            0,
            10,
        )

        return pd.DataFrame(
            {
                "week": weeks,
                "pain": pain,
                "swelling": swelling,
                "activity": activity,
            }
        )

    def build_intake_impact(self, user_input: Sequence[float]) -> pd.DataFrame:
        """Build 30-day intake impact curves for beetroot and fenugreek."""
        x = self._sanitize_user_input(user_input)
        days = np.arange(0, 31)

        beetroot_active = (days <= int(x[6])).astype(float)
        fenugreek_active = (days <= int(x[8])).astype(float)

        beetroot_effect = x[7] * (1 - np.exp(-0.12 * days)) * beetroot_active
        fenugreek_effect = x[9] * (1 - np.exp(-0.09 * days)) * fenugreek_active
        combined_effect = beetroot_effect + fenugreek_effect

        return pd.DataFrame(
            {
                "day": days,
                "beetroot_effect": beetroot_effect,
                "fenugreek_effect": fenugreek_effect,
                "combined_effect": combined_effect,
            }
        )

    def create_recovery_chart(self, user_input: Sequence[float], recovery_weeks: float) -> None:
        """Create and save the recovery progression chart."""
        chart_df = self.build_recovery_trajectory(user_input, recovery_weeks)

        plt.figure(figsize=(12, 6))
        plt.plot(chart_df["week"], chart_df["pain"], color="#C0392B", linewidth=2.8, marker="o", label="Pain")
        plt.plot(
            chart_df["week"],
            chart_df["swelling"],
            color="#E67E22",
            linewidth=2.8,
            marker="o",
            label="Swelling",
        )
        plt.plot(
            chart_df["week"],
            chart_df["activity"],
            color="#1E8449",
            linewidth=2.8,
            marker="o",
            label="Activity",
        )

        plt.xlabel("Weeks")
        plt.ylabel("Symptom Scale (0-10)")
        plt.title("Projected Symptom Trend During Recovery")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output_chart.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Recovery chart saved as 'output_chart.png'")

    def create_intake_impact_chart(self, user_input: Sequence[float]) -> None:
        """Create and save supplement intake impact charts."""
        chart_df = self.build_intake_impact(user_input)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(chart_df["day"], chart_df["beetroot_effect"], color="#B03A2E", linewidth=2.5)
        plt.fill_between(chart_df["day"], chart_df["beetroot_effect"], alpha=0.25, color="#F1948A")
        plt.title("Beetroot Intake Impact (30 Days)")
        plt.xlabel("Day")
        plt.ylabel("Relative Effect")
        plt.grid(True, alpha=0.25)

        plt.subplot(1, 2, 2)
        plt.plot(chart_df["day"], chart_df["fenugreek_effect"], color="#196F3D", linewidth=2.5)
        plt.fill_between(chart_df["day"], chart_df["fenugreek_effect"], alpha=0.25, color="#82E0AA")
        plt.title("Fenugreek Intake Impact (30 Days)")
        plt.xlabel("Day")
        plt.ylabel("Relative Effect")
        plt.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.savefig("intake_impact.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Intake impact chart saved as 'intake_impact.png'")

    def generate_rule_based_narrative(
        self,
        user_input: Sequence[float],
        risk_level: str,
        recovery_weeks: float,
        suggestions: Sequence[str],
        comparison: Dict[str, Dict[str, Any]],
    ) -> str:
        """Fallback narrative when Gemini is unavailable."""
        current_weeks = comparison["current"]["recovery_weeks"]
        no_supp_weeks = comparison["no_supplements"]["recovery_weeks"]
        optimal_weeks = comparison["optimal_supplements"]["recovery_weeks"]

        gain_vs_none = no_supp_weeks - current_weeks
        gain_vs_current = current_weeks - optimal_weeks

        lines = [
            "Model interpretation:",
            f"- Estimated risk category: {risk_level}.",
            f"- Estimated recovery duration: {recovery_weeks:.1f} weeks.",
            f"- Current supplement routine is projected to improve recovery by {gain_vs_none:.1f} weeks versus no supplements.",
        ]

        if gain_vs_current > 0.05:
            lines.append(
                f"- A better regimen was found, with an additional projected improvement of {gain_vs_current:.1f} weeks."
            )
        else:
            lines.append("- Your current regimen is already close to the best model-estimated regimen.")

        lines.append("- Suggested next steps:")
        for idx, suggestion in enumerate(suggestions[:4], start=1):
            lines.append(f"  {idx}. {suggestion}")

        lines.append(
            "- This output is decision support from a synthetic-data model and is not a clinical diagnosis."
        )
        return "\n".join(lines)

    def _extract_gemini_text(self, payload: Dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            return ""
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return ""
        text = parts[0].get("text", "")
        return text.strip()

    def _build_gemini_prompt(
        self,
        user_input: Sequence[float],
        risk_level: str,
        recovery_weeks: float,
        suggestions: Sequence[str],
        comparison: Dict[str, Dict[str, Any]],
    ) -> str:
        x = self._sanitize_user_input(user_input)
        return f"""
You are assisting with a varicose vein recovery prediction project.
Write a concise, practical interpretation in plain English for a non-technical user.

Patient profile:
- Age: {x[0]:.1f}
- BMI: {x[1]:.1f}
- Family history: {"Yes" if x[2] >= 0.5 else "No"}
- Pain: {x[3]:.1f}/10
- Swelling: {x[4]:.1f}/10
- Activity: {x[5]:.1f}/10
- Beetroot intake: {x[7]:.1f} g/day for {x[6]:.0f} days/month
- Fenugreek intake: {x[9]:.1f} g/day for {x[8]:.0f} days/month

Predictions:
- Risk level: {risk_level}
- Recovery estimate: {recovery_weeks:.1f} weeks
- Current regimen recovery: {comparison["current"]["recovery_weeks"]:.1f} weeks
- No supplements recovery: {comparison["no_supplements"]["recovery_weeks"]:.1f} weeks
- Optimized regimen recovery: {comparison["optimal_supplements"]["recovery_weeks"]:.1f} weeks

Suggestions:
{chr(10).join(f"- {s}" for s in suggestions[:6])}

Response requirements:
- Start with a short summary paragraph.
- Then provide exactly 4 bullet recommendations.
- End with one safety line that this is not a diagnosis.
- Keep total length under 220 words.
""".strip()

    def generate_ai_narrative(
        self,
        user_input: Sequence[float],
        risk_level: str,
        recovery_weeks: float,
        suggestions: Sequence[str],
        comparison: Dict[str, Dict[str, Any]],
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Generate AI narrative with Gemini when an API key is available.
        Returns (narrative_text, source) where source is 'gemini' or a fallback reason.
        """
        fallback_text = self.generate_rule_based_narrative(
            user_input=user_input,
            risk_level=risk_level,
            recovery_weeks=recovery_weeks,
            suggestions=suggestions,
            comparison=comparison,
        )

        key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not key:
            return fallback_text, "fallback_no_key"

        if requests is None:
            return fallback_text, "fallback_requests_missing"

        chosen_model = (model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()
        prompt = self._build_gemini_prompt(
            user_input=user_input,
            risk_level=risk_level,
            recovery_weeks=recovery_weeks,
            suggestions=suggestions,
            comparison=comparison,
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 420,
            },
        }

        candidate_models = []
        for candidate in [
            chosen_model,
            os.getenv("GEMINI_FALLBACK_MODEL", "").strip(),
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-flash-latest",
        ]:
            if not candidate:
                continue
            normalized = candidate.replace("models/", "").strip()
            if normalized and normalized not in candidate_models:
                candidate_models.append(normalized)

        saw_model_not_found = False
        for candidate_model in candidate_models:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{candidate_model}:generateContent?key={key}"
            )
            try:
                response = requests.post(url, json=payload, timeout=25)
                if response.status_code == 404:
                    saw_model_not_found = True
                    continue
                response.raise_for_status()
                generated_text = self._extract_gemini_text(response.json())
                if generated_text:
                    return generated_text, "gemini"
            except Exception:
                continue

        if saw_model_not_found:
            return fallback_text, "fallback_model_not_found"
        return fallback_text, "fallback_api_error"

    def build_results_summary_text(
        self,
        user_input: Sequence[float],
        risk_level: str,
        recovery_weeks: float,
        suggestions: Sequence[str],
        comparison: Dict[str, Dict[str, Any]],
        narrative_text: Optional[str] = None,
        narrative_source: str = "rule_based",
    ) -> str:
        x = self._sanitize_user_input(user_input)
        narrative = narrative_text or self.generate_rule_based_narrative(
            user_input=x,
            risk_level=risk_level,
            recovery_weeks=recovery_weeks,
            suggestions=suggestions,
            comparison=comparison,
        )

        lines = [
            "=== VARICOSE VEIN RECOVERY ANALYSIS SUMMARY ===",
            "",
            "Patient Profile:",
            f"- Age: {x[0]:.1f} years",
            f"- BMI: {x[1]:.1f}",
            f"- Family History: {'Yes' if x[2] >= 0.5 else 'No'}",
            f"- Pain Level: {x[3]:.1f}/10",
            f"- Swelling Level: {x[4]:.1f}/10",
            f"- Activity Level: {x[5]:.1f}/10",
            "",
            "Current Supplement Intake:",
            f"- Beetroot: {x[7]:.1f} g/day for {x[6]:.0f} days/month",
            f"- Fenugreek: {x[9]:.1f} g/day for {x[8]:.0f} days/month",
            "",
            "Predictions:",
            f"- Risk Level: {risk_level}",
            f"- Estimated Recovery Time: {recovery_weeks:.1f} weeks",
            "",
            "Supplement Comparison:",
            f"- Current regimen: {comparison['current']['risk']} risk, {comparison['current']['recovery_weeks']:.1f} weeks",
            f"- No supplements: {comparison['no_supplements']['risk']} risk, {comparison['no_supplements']['recovery_weeks']:.1f} weeks",
            f"- Optimized supplements: {comparison['optimal_supplements']['risk']} risk, {comparison['optimal_supplements']['recovery_weeks']:.1f} weeks",
            "",
            "Top Recommendations:",
        ]

        for idx, suggestion in enumerate(suggestions[:6], start=1):
            lines.append(f"{idx}. {suggestion}")

        lines.extend(
            [
                "",
                f"Narrative source: {narrative_source}",
                "Narrative:",
                narrative,
            ]
        )

        if self.metrics_:
            cls_metrics = self.metrics_.get("classification", {})
            reg_metrics = self.metrics_.get("regression", {})
            auc_value = cls_metrics.get("auc_roc_ovr_weighted")
            auc_display = f"{auc_value:.3f}" if auc_value is not None else "NA"

            lines.extend(
                [
                    "",
                    "Model Validation:",
                    f"- Accuracy: {cls_metrics.get('accuracy', float('nan')):.3f}",
                    f"- Precision (weighted): {cls_metrics.get('precision_weighted', float('nan')):.3f}",
                    f"- Recall (weighted): {cls_metrics.get('recall_weighted', float('nan')):.3f}",
                    f"- F1 (weighted): {cls_metrics.get('f1_weighted', float('nan')):.3f}",
                    f"- AUC-ROC OvR (weighted): {auc_display}",
                    f"- Recovery R2: {reg_metrics.get('r2', float('nan')):.3f}",
                    f"- Recovery MSE: {reg_metrics.get('mse', float('nan')):.3f}",
                ]
            )

        lines.extend(
            [
                "",
                "Generated by Varicose Vein Recovery ML Model",
            ]
        )
        return "\n".join(lines)

    def save_results_summary(
        self,
        user_input: Sequence[float],
        risk_level: str,
        recovery_weeks: float,
        suggestions: Sequence[str],
        comparison: Dict[str, Dict[str, Any]],
        narrative_text: Optional[str] = None,
        narrative_source: str = "rule_based",
        file_path: str = "results_summary.txt",
    ) -> None:
        summary_text = self.build_results_summary_text(
            user_input=user_input,
            risk_level=risk_level,
            recovery_weeks=recovery_weeks,
            suggestions=suggestions,
            comparison=comparison,
            narrative_text=narrative_text,
            narrative_source=narrative_source,
        )
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(summary_text)
        print(f"Results summary saved as '{file_path}'")


def _prompt_float(prompt: str, min_value: float, max_value: float) -> float:
    while True:
        try:
            value = float(input(prompt))
            if value < min_value or value > max_value:
                print(f"Enter a value between {min_value} and {max_value}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Enter a numeric value.")


def get_user_input() -> List[float]:
    """Collect and validate user input from terminal."""
    print("\n=== VARICOSE VEIN RECOVERY PREDICTOR ===")
    print("Please enter the following information:")

    age = _prompt_float("Age (20-80 years): ", 20, 80)
    bmi = _prompt_float("BMI (15-45): ", 15, 45)
    family_history = _prompt_float("Family history (0=No, 1=Yes): ", 0, 1)
    pain_level = _prompt_float("Pain level (0-10): ", 0, 10)
    swelling = _prompt_float("Swelling level (0-10): ", 0, 10)
    activity_level = _prompt_float("Activity level (0-10): ", 0, 10)
    beetroot_days_30 = _prompt_float("Beetroot intake days in last 30 days (0-30): ", 0, 30)
    beetroot_grams_day = _prompt_float("Beetroot grams per day (0-30): ", 0, 30)
    fenugreek_days_30 = _prompt_float("Fenugreek intake days in last 30 days (0-30): ", 0, 30)
    fenugreek_grams_day = _prompt_float("Fenugreek grams per day (0-20): ", 0, 20)

    return [
        age,
        bmi,
        family_history,
        pain_level,
        swelling,
        activity_level,
        beetroot_days_30,
        beetroot_grams_day,
        fenugreek_days_30,
        fenugreek_grams_day,
    ]


def main() -> None:
    """Main function for CLI workflow."""
    print("Initializing Varicose Vein Recovery ML Model...")
    model = VaricoseVeinMLModel()

    def train_fresh_models() -> None:
        print("Generating synthetic training data...")
        data = model.generate_synthetic_data(8000)
        print("Training ML models...")
        model.train_models(
            data=data,
            enable_tuning=True,
            save_artifacts=True,
            verbose=True,
        )

    force_retrain = os.getenv("FORCE_RETRAIN", "0").strip() == "1"
    if model.artifacts_exist() and not force_retrain:
        print("Loading existing model artifacts...")
        try:
            model.load_artifacts()
        except Exception as exc:
            print(f"Artifact load failed ({type(exc).__name__}). Re-training models...")
            train_fresh_models()
    else:
        train_fresh_models()

    user_input = get_user_input()

    print("\nRunning predictions...")
    risk_level, recovery_weeks, risk_probs = model.predict_single_case(user_input, return_probabilities=True)
    suggestions = model.generate_suggestions(user_input, risk_level, recovery_weeks)
    comparison = model.compare_supplement_effect(user_input)

    narrative_text, narrative_source = model.generate_ai_narrative(
        user_input=user_input,
        risk_level=risk_level,
        recovery_weeks=recovery_weeks,
        suggestions=suggestions,
        comparison=comparison,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Risk Level: {risk_level}")
    print(f"Estimated Recovery Time: {recovery_weeks:.1f} weeks")
    print("Risk Probabilities:")
    for label in ["Low", "Moderate", "High"]:
        print(f"- {label}: {risk_probs.get(label, 0.0):.1%}")

    print("\nTop Recommendations:")
    for idx, suggestion in enumerate(suggestions, start=1):
        print(f"{idx}. {suggestion}")

    print("\nSupplement Comparison:")
    for scenario in ["current", "no_supplements", "optimal_supplements"]:
        row = comparison[scenario]
        scenario_name = scenario.replace("_", " ").title()
        print(f"- {scenario_name}: {row['risk']} risk, {row['recovery_weeks']:.1f} weeks")

    print(f"\nNarrative source: {narrative_source}")

    print("\nGenerating charts and saving results...")
    model.create_recovery_chart(user_input, recovery_weeks)
    model.create_intake_impact_chart(user_input)
    model.save_results_summary(
        user_input=user_input,
        risk_level=risk_level,
        recovery_weeks=recovery_weeks,
        suggestions=suggestions,
        comparison=comparison,
        narrative_text=narrative_text,
        narrative_source=narrative_source,
    )

    print("\nAnalysis complete. Files saved:")
    print("- output_chart.png")
    print("- intake_impact.png")
    print("- results_summary.txt")
    print("- risk_classifier.pkl")
    print("- recovery_regressor.pkl")
    print("- scaler.pkl")


if __name__ == "__main__":
    main()
