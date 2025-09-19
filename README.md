Varicose Vein Recovery Model

This project shows how a simple machine learning model can simulate the effect of natural supplements (beetroot + fenugreek) on varicose vein recovery. It’s not based on real clinical data — the goal is just to demonstrate how ML can be used in health-related simulations.

What it does

Predicts risk level (Low / Moderate / High) using a Random Forest model
Estimates recovery time in weeks with Gradient Boosting
Compares scenarios: no supplements vs current intake vs optimal intake
Generates simple charts to show progress (pain ↓, swelling ↓, activity ↑)

Features
Terminal-based input: You just run the script and enter the patient's info right into the terminal — no need for a frontend.

Models Used

Random Forest Classifier → predicts recovery risk (Low / Moderate / High)

Gradient Boosting Regressor → estimates recovery time in weeks

Both models are trained on a synthetic dataset (~8000 samples) that mimics realistic recovery patterns. No medical data is required.
Charts & Visuals:

Visuals & Outputs

output_chart.png → shows how pain, swelling, and activity levels change over time
intake_impact.png → compares recovery speed with vs. without supplements
results_summary.txt → simple text summary with risk, recovery time, and suggestions
Models (.pkl files) are also saved so you don’t need to retrain every time.

How It Works

A synthetic dataset is generated with patient-like profiles (age, BMI, symptoms, supplement intake, etc.).

Models are trained:

Random Forest → risk level

Gradient Boosting → recovery time

You enter patient details in the terminal.

The program predicts recovery and compares three cases:

No supplements

Current intake

Optimal intake

Results are shown in charts and a summary file.

Getting Started
Steps to set up and run the project.

Clone the repository:

git clone https://github.com/AmanDevNet/Varicose-Vein-Recovery-Model.git

cd Varicose-Vein-Recovery-Model

Install the required Python packages:

pip install numpy pandas scikit-learn matplotlib seaborn joblib

Usage
To run the model and get predictions:

Execute the main script:

python varicose_vein_model.py

**Output**

Recovery chart saved as 'output_chart.png', 
Intake impact chart saved as 'intake_impact.png', 
Results summary saved as 'results_summary.txt'


Output Files

output_chart.png → recovery progression (pain ↓, swelling ↓, activity ↑)
<img width="4176" height="2074" alt="output_chart" src="https://github.com/user-attachments/assets/c4f5a79e-a96a-43a8-8f17-1cbcf3636cd8" />

intake_impact.png → effect of supplements on recovery speed
<img width="3569" height="1765" alt="intake_impact" src="https://github.com/user-attachments/assets/a29eddc7-995b-47f2-ad4d-927ecdbbb7f5" />

results_summary.txt → text summary with predictions

risk_classifier.pkl, recovery_regressor.pkl, scaler.pkl → saved models and preprocessor

recovery_regressor.pkl: The saved trained Gradient Boosting Regressor model.

scaler.pkl: The saved data scaler used for preprocessing.

Contributions are welcome — feel free to suggest improvements or add features!

Author Aman Sharma 
LinkedIn :- www.linkedin.com/in/aman-sharma-842b66318
