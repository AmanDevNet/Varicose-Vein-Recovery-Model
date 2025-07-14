import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class VaricoseVeinMLModel:
    def __init__(self):
        self.risk_classifier = None
        self.recovery_regressor = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'age', 'bmi', 'family_history', 'pain_level', 'swelling',
            'activity_level', 'beetroot_days_30', 'beetroot_grams_day',
            'fenugreek_days_30', 'fenugreek_grams_day'
        ]
        
    def generate_synthetic_data(self, n_samples=8000):
        """Generate synthetic dataset for varicose vein recovery"""
        np.random.seed(42)
        
        # Generate base features
        age = np.random.normal(50, 15, n_samples)
        age = np.clip(age, 20, 80)
        
        bmi = np.random.normal(26, 4, n_samples)
        bmi = np.clip(bmi, 18, 40)
        
        family_history = np.random.binomial(1, 0.3, n_samples)
        
        # Generate symptoms (correlated with age and BMI)
        pain_level = np.random.normal(5 + age/20 + bmi/10, 2, n_samples)
        pain_level = np.clip(pain_level, 0, 10)
        
        swelling = np.random.normal(4 + age/25 + bmi/8, 2, n_samples)
        swelling = np.clip(swelling, 0, 10)
        
        activity_level = np.random.normal(6 - age/30 - bmi/12, 2, n_samples)
        activity_level = np.clip(activity_level, 0, 10)
        
        # Generate supplement intake (some correlation with health awareness)
        beetroot_days_30 = np.random.poisson(15, n_samples)
        beetroot_days_30 = np.clip(beetroot_days_30, 0, 30)
        
        beetroot_grams_day = np.random.normal(10, 5, n_samples)
        beetroot_grams_day = np.clip(beetroot_grams_day, 0, 30)
        
        fenugreek_days_30 = np.random.poisson(12, n_samples)
        fenugreek_days_30 = np.clip(fenugreek_days_30, 0, 30)
        
        fenugreek_grams_day = np.random.normal(8, 4, n_samples)
        fenugreek_grams_day = np.clip(fenugreek_grams_day, 0, 20)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'bmi': bmi,
            'family_history': family_history,
            'pain_level': pain_level,
            'swelling': swelling,
            'activity_level': activity_level,
            'beetroot_days_30': beetroot_days_30,
            'beetroot_grams_day': beetroot_grams_day,
            'fenugreek_days_30': fenugreek_days_30,
            'fenugreek_grams_day': fenugreek_grams_day
        })
        
        # Generate target variables
        # Risk level (0: Low, 1: Moderate, 2: High)
        risk_score = (
            0.3 * (data['age'] - 20) / 60 +
            0.2 * (data['bmi'] - 18) / 22 +
            0.2 * data['family_history'] +
            0.15 * data['pain_level'] / 10 +
            0.15 * data['swelling'] / 10 -
            0.1 * (data['beetroot_days_30'] * data['beetroot_grams_day']) / 300 -
            0.1 * (data['fenugreek_days_30'] * data['fenugreek_grams_day']) / 200
        )
        
        risk_level = np.where(risk_score < 0.3, 0, 
                             np.where(risk_score < 0.7, 1, 2))
        
        # Recovery time in weeks
        recovery_weeks = (
            8 + 
            0.2 * (data['age'] - 50) / 10 +
            0.15 * (data['bmi'] - 25) / 5 +
            0.1 * data['family_history'] * 4 +
            0.2 * data['pain_level'] +
            0.2 * data['swelling'] -
            0.1 * data['activity_level'] -
            0.15 * (data['beetroot_days_30'] * data['beetroot_grams_day']) / 100 -
            0.1 * (data['fenugreek_days_30'] * data['fenugreek_grams_day']) / 60 +
            np.random.normal(0, 1, n_samples)
        )
        recovery_weeks = np.clip(recovery_weeks, 2, 20)
        
        data['risk_level'] = risk_level
        data['recovery_weeks'] = recovery_weeks
        
        return data
    
    def train_models(self, data):
        """Train both classification and regression models"""
        X = data[self.feature_columns]
        y_risk = data['risk_level']
        y_recovery = data['recovery_weeks']
        
        # Split data
        X_train, X_test, y_risk_train, y_risk_test, y_recovery_train, y_recovery_test = train_test_split(
            X, y_risk, y_recovery, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train risk classifier
        self.risk_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.risk_classifier.fit(X_train_scaled, y_risk_train)
        
        # Train recovery regressor
        self.recovery_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.recovery_regressor.fit(X_train_scaled, y_recovery_train)
        
        # Evaluate models
        risk_pred = self.risk_classifier.predict(X_test_scaled)
        recovery_pred = self.recovery_regressor.predict(X_test_scaled)
        
        print("=== MODEL PERFORMANCE ===")
        print(f"Risk Classification Accuracy: {self.risk_classifier.score(X_test_scaled, y_risk_test):.3f}")
        print(f"Recovery Prediction R²: {r2_score(y_recovery_test, recovery_pred):.3f}")
        print(f"Recovery Prediction MSE: {mean_squared_error(y_recovery_test, recovery_pred):.3f}")
        
        # Save models
        joblib.dump(self.risk_classifier, 'risk_classifier.pkl')
        joblib.dump(self.recovery_regressor, 'recovery_regressor.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        return X_test_scaled, y_risk_test, y_recovery_test, risk_pred, recovery_pred
    
    def predict_single_case(self, user_input):
        """Make prediction for a single case"""
        # Convert input to DataFrame
        input_df = pd.DataFrame([user_input], columns=self.feature_columns)
        input_scaled = self.scaler.transform(input_df)
        
        # Make predictions
        risk_pred = self.risk_classifier.predict(input_scaled)[0]
        recovery_pred = self.recovery_regressor.predict(input_scaled)[0]
        
        risk_labels = ['Low', 'Moderate', 'High']
        risk_level = risk_labels[risk_pred]
        
        return risk_level, recovery_pred
    
    def generate_suggestions(self, user_input, risk_level, recovery_weeks):
        """Generate personalized suggestions"""
        suggestions = []
        
        # Age-based suggestions
        if user_input[0] > 60:
            suggestions.append("Consider gentle exercise routines suitable for your age group")
        
        # BMI-based suggestions
        if user_input[1] > 25:
            suggestions.append("Weight management can significantly improve varicose vein symptoms")
        
        # Pain/Swelling suggestions
        if user_input[3] > 6 or user_input[4] > 6:
            suggestions.append("Elevate legs regularly and consider compression stockings")
        
        # Activity level suggestions
        if user_input[5] < 5:
            suggestions.append("Gradually increase physical activity - walking is excellent for circulation")
        
        # Supplement suggestions
        if user_input[6] < 20 or user_input[7] < 8:
            suggestions.append("Increase beetroot intake to 10-15g daily for better nitric oxide production")
        
        if user_input[8] < 15 or user_input[9] < 6:
            suggestions.append("Consider taking fenugreek 8-12g daily for anti-inflammatory benefits")
        
        # Risk-specific suggestions
        if risk_level == 'High':
            suggestions.append("Consult with a vascular specialist for comprehensive treatment plan")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def compare_supplement_effect(self, user_input):
        """Compare predictions with and without supplements"""
        # Original prediction
        original_risk, original_recovery = self.predict_single_case(user_input)
        
        # Prediction without supplements
        no_supplement_input = user_input.copy()
        no_supplement_input[6] = 0  # beetroot_days_30
        no_supplement_input[7] = 0  # beetroot_grams_day
        no_supplement_input[8] = 0  # fenugreek_days_30
        no_supplement_input[9] = 0  # fenugreek_grams_day
        
        no_supp_risk, no_supp_recovery = self.predict_single_case(no_supplement_input)
        
        # Prediction with optimal supplements
        optimal_input = user_input.copy()
        optimal_input[6] = 25  # beetroot_days_30
        optimal_input[7] = 15  # beetroot_grams_day
        optimal_input[8] = 25  # fenugreek_days_30
        optimal_input[9] = 12  # fenugreek_grams_day
        
        optimal_risk, optimal_recovery = self.predict_single_case(optimal_input)
        
        return {
            'current': (original_risk, original_recovery),
            'no_supplements': (no_supp_risk, no_supp_recovery),
            'optimal_supplements': (optimal_risk, optimal_recovery)
        }
    
    def create_recovery_chart(self, user_input, recovery_weeks):
        """Create recovery progression chart with black background and save as output_chart.png"""
        import matplotlib.pyplot as plt
        import numpy as np

        weeks = np.arange(0, int(recovery_weeks) + 1)
        initial_pain = user_input[3]
        initial_swelling = user_input[4]
        initial_activity = user_input[5]

        # Recovery curves
        pain_progression = initial_pain * np.exp(-0.15 * weeks)
        swelling_progression = initial_swelling * np.exp(-0.12 * weeks)
        activity_progression = initial_activity + (10 - initial_activity) * (1 - np.exp(-0.1 * weeks))

        plt.figure(figsize=(14, 7))
        ax = plt.gca()
        # Set black background
        ax.set_facecolor('black')
        plt.gcf().patch.set_facecolor('black')
        # Plot lines
        plt.plot(weeks, pain_progression, color='red', linewidth=3, marker='o', label='Pain ↓')
        plt.plot(weeks, swelling_progression, color='orange', linewidth=3, marker='o', label='Swelling ↓')
        plt.plot(weeks, activity_progression, color='lime', linewidth=3, marker='o', label='Activity ↑')
        # Axes and grid
        plt.xlabel('Weeks', fontsize=16, color='white')
        plt.ylabel('Symptom Level (0-10)', fontsize=16, color='white')
        plt.title('Symptom Improvement Over Time (Beetroot + Fenugreek)', fontsize=18, color='white', pad=20)
        plt.grid(True, alpha=0.3, color='white')
        plt.legend(loc='best', fontsize=14, facecolor='black', edgecolor='white', labelcolor='white')
        # Set tick params
        ax.tick_params(axis='x', colors='white', labelsize=13)
        ax.tick_params(axis='y', colors='white', labelsize=13)
        # Set spines
        for spine in ax.spines.values():
            spine.set_color('white')
        # Save chart
        plt.tight_layout()
        plt.savefig('output_chart.png', dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        print("✅ Recovery chart saved as 'output_chart.png'")
    
    def create_intake_impact_chart(self, user_input):
        """Create supplement intake impact chart"""
        days = np.arange(0, 31)
        
        # Simulate beetroot and fenugreek effects over time
        beetroot_effect = np.zeros(31)
        fenugreek_effect = np.zeros(31)
        
        # Add effect on days when supplements are taken
        beetroot_days = int(user_input[6])
        fenugreek_days = int(user_input[8])
        
        for day in range(31):
            if day < beetroot_days:
                beetroot_effect[day] = user_input[7] * (1 - np.exp(-0.1 * day))
            if day < fenugreek_days:
                fenugreek_effect[day] = user_input[9] * (1 - np.exp(-0.08 * day))
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(days, beetroot_effect, 'r-', linewidth=3, label='Beetroot Effect')
        plt.fill_between(days, beetroot_effect, alpha=0.3, color='red')
        plt.xlabel('Days')
        plt.ylabel('Cumulative Effect')
        plt.title('Beetroot Impact Over 30 Days')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(days, fenugreek_effect, 'g-', linewidth=3, label='Fenugreek Effect')
        plt.fill_between(days, fenugreek_effect, alpha=0.3, color='green')
        plt.xlabel('Days')
        plt.ylabel('Cumulative Effect')
        plt.title('Fenugreek Impact Over 30 Days')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('intake_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Intake impact chart saved as 'intake_impact.png'")
    
    def save_results_summary(self, user_input, risk_level, recovery_weeks, suggestions, comparison):
        """Save results summary to text file"""
        summary = f"""
=== VARICOSE VEIN RECOVERY ANALYSIS SUMMARY ===

Patient Profile:
- Age: {user_input[0]:.1f} years
- BMI: {user_input[1]:.1f}
- Family History: {'Yes' if user_input[2] else 'No'}
- Pain Level: {user_input[3]:.1f}/10
- Swelling Level: {user_input[4]:.1f}/10
- Activity Level: {user_input[5]:.1f}/10

Current Supplement Intake:
- Beetroot: {user_input[7]:.1f}g/day for {user_input[6]:.0f} days/month
- Fenugreek: {user_input[9]:.1f}g/day for {user_input[8]:.0f} days/month

PREDICTIONS:
- Risk Level: {risk_level}
- Estimated Recovery Time: {recovery_weeks:.1f} weeks

SUPPLEMENT COMPARISON:
- Current regimen: {comparison['current'][0]} risk, {comparison['current'][1]:.1f} weeks recovery
- No supplements: {comparison['no_supplements'][0]} risk, {comparison['no_supplements'][1]:.1f} weeks recovery
- Optimal supplements: {comparison['optimal_supplements'][0]} risk, {comparison['optimal_supplements'][1]:.1f} weeks recovery

TOP RECOMMENDATIONS:
"""
        
        for i, suggestion in enumerate(suggestions, 1):
            summary += f"{i}. {suggestion}\n"
        
        summary += """
RESEARCH NARRATIVE:
This ML model simulates how beetroot & fenugreek intake affects varicose vein recovery. 
The results show that regular intake can lower risk levels and reduce recovery time, 
which supports the formulation of a combined syrup.

Generated by Varicose Vein Recovery ML Model
"""
        
        with open('results_summary.txt', 'w') as f:
            f.write(summary)
        
        print("✅ Results summary saved as 'results_summary.txt'")

def get_user_input():
    """Get user input from terminal"""
    print("\n=== VARICOSE VEIN RECOVERY PREDICTOR ===")
    print("Please enter the following information:")
    
    try:
        age = float(input("Age (years): "))
        bmi = float(input("BMI: "))
        family_history = int(input("Family history of varicose veins (0=No, 1=Yes): "))
        pain_level = float(input("Pain level (0-10): "))
        swelling = float(input("Swelling level (0-10): "))
        activity_level = float(input("Activity level (0-10): "))
        beetroot_days_30 = float(input("Beetroot intake days in last 30 days: "))
        beetroot_grams_day = float(input("Beetroot grams per day: "))
        fenugreek_days_30 = float(input("Fenugreek intake days in last 30 days: "))
        fenugreek_grams_day = float(input("Fenugreek grams per day: "))
        
        return [age, bmi, family_history, pain_level, swelling, activity_level,
                beetroot_days_30, beetroot_grams_day, fenugreek_days_30, fenugreek_grams_day]
    
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return get_user_input()

def main():
    """Main function to run the ML model"""
    print("🧪 Initializing Varicose Vein Recovery ML Model...")
    
    # Initialize model
    model = VaricoseVeinMLModel()
    
    # Generate synthetic data
    print("📊 Generating synthetic training data...")
    data = model.generate_synthetic_data(8000)
    
    # Train models
    print("🤖 Training ML models...")
    model.train_models(data)
    
    # Get user input
    user_input = get_user_input()
    
    # Make predictions
    print("\n🔮 Making predictions...")
    risk_level, recovery_weeks = model.predict_single_case(user_input)
    
    # Generate suggestions
    suggestions = model.generate_suggestions(user_input, risk_level, recovery_weeks)
    
    # Compare supplement effects
    comparison = model.compare_supplement_effect(user_input)
    
    # Display results
    print("\n" + "="*50)
    print("📋 RESULTS")
    print("="*50)
    print(f"Risk Level: {risk_level}")
    print(f"Estimated Recovery Time: {recovery_weeks:.1f} weeks")
    print("\nTop Recommendations:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    print("\n📊 Supplement Comparison:")
    print(f"Current regimen: {comparison['current'][0]} risk, {comparison['current'][1]:.1f} weeks")
    print(f"No supplements: {comparison['no_supplements'][0]} risk, {comparison['no_supplements'][1]:.1f} weeks")
    print(f"Optimal supplements: {comparison['optimal_supplements'][0]} risk, {comparison['optimal_supplements'][1]:.1f} weeks")
    
    # Generate charts and save results
    print("\n📈 Generating charts and saving results...")
    model.create_recovery_chart(user_input, recovery_weeks)
    model.create_intake_impact_chart(user_input)
    model.save_results_summary(user_input, risk_level, recovery_weeks, suggestions, comparison)
    
    print("\n✅ Analysis complete! Files saved:")
    print("   - output_chart.png")
    print("   - intake_impact.png")
    print("   - results_summary.txt")
    print("   - Model files: risk_classifier.pkl, recovery_regressor.pkl, scaler.pkl")

if __name__ == "__main__":
    main()