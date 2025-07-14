Varicose Vein Recovery Model

This project uses machine learning to demonstrate how a multivitamin syrup made from beetroot and fenugreek might help in the recovery of varicose veins. The idea is to simulate and show how these natural ingredients could improve symptoms and reduce recovery time — even without needing real clinical data.

Project Goal
The main purpose of this model is to simulate the effectiveness of a combined syrup (beetroot + fenugreek) for treating varicose veins. It helps visualize improvements in symptoms like pain, swelling, and activity level, as well as estimate how long recovery may take — with and without supplements.

Features
Terminal-based input: You just run the script and enter the patient's info right into the terminal — no need for a frontend.

Two machine learning models:

RandomForestClassifier to predict risk level: Low / Moderate / High

GradientBoostingRegressor to predict recovery time (in weeks)

Synthetic dataset generation: The model is trained on realistic simulated data (around 8000 entries), so you don’t need any actual medical dataset.

Charts & Visuals:

output_chart.png: Shows improvement in symptoms over time (pain ↓, swelling ↓, activity ↑)

intake_impact.png: Compares supplement intake vs recovery speed

Summary Output:

Recovery risk and time

Personalized suggestions

Comparison of "Current intake" vs "No supplements" vs "Optimal supplements"

Reusable model: Trained models (.pkl files) are saved so you can reuse them later without retraining.

How It Works
The project simulates the progression and recovery of varicose veins based on various input factors.

Data Generation: A synthetic dataset is created, mimicking realistic patient profiles and recovery patterns, where beetroot and fenugreek intake are designed to positively influence outcomes.

Model Training:

The RandomForestClassifier learns to categorize risk levels (Low, Moderate, High) based on the input features.

The GradientBoostingRegressor learns to predict the recovery time in weeks.

User Input: You provide specific patient parameters (age, BMI, symptoms, supplement intake) via the terminal.

Prediction: The trained models use your input to predict the risk level and estimated recovery time for the simulated patient.

Supplement Comparison: A key feature where the model compares the predicted outcome for your input against a scenario with "No supplements" and an "Optimal supplements" scenario, directly demonstrating the benefits.

Visualization & Summary:  Charts are generated to visually represent the recovery progression, and a summary text file captures all the results.

Getting Started
Follow these steps to set up and run the project.

Prerequisites
Python 3.x

pip (Python package installer)

Installation
Clone the repository:

git clone https://github.com/AmanDevNet/Varicose-Vein-Recovery-Model.git
cd Varicose-Vein-Recovery-Model

Install the required Python packages:

pip install numpy pandas scikit-learn matplotlib seaborn joblib

Usage
To run the model and get predictions:

Execute the main script:

python varicose_vein_model.py

Follow the prompts: The script will ask you to enter various patient details in the terminal.

🧪 Initializing Varicose Vein Recovery ML Model...
📊 Generating synthetic training data...
🤖 Training ML models...
=== MODEL PERFORMANCE ===
Risk Classification Accuracy: 0.952
Recovery Prediction R²: 0.338
Recovery Prediction MSE: 0.993

=== VARICOSE VEIN RECOVERY PREDICTOR ===
Please enter the following information:
Age (years): 44
BMI: 24.5
Family history of varicose veins (0=No, 1=Yes): 0
Pain level (0-10): 7
Swelling level (0-10): 5
Activity level (0-10): 8
Beetroot intake days in last 30 days: 27
Beetroot grams per day: 9
Fenugreek intake days in last 30 days: 27
Fenugreek grams per day: 9

🔮 Making predictions...

View the results: After entering all inputs, the script will print the predictions to the terminal and save the output files.

==================================================
📋 RESULTS
==================================================
Risk Level: Low
Estimated Recovery Time: 9.8 weeks

Top Recommendations:
1. Elevate legs regularly and consider compression stockings

📊 Supplement Comparison:
Current regimen: Low risk, 9.8 weeks
No supplements: Moderate risk, 9.4 weeks
Optimal supplements: Low risk, 8.4 weeks

📈 Generating charts and saving results...
✅ Recovery chart saved as 'output_chart.png'
✅ Intake impact chart saved as 'intake_impact.png'
✅ Results summary saved as 'results_summary.txt'

✅ Analysis complete! Files saved:
    - output_chart.png
    - intake_impact.png
    - results_summary.txt
    - Model files: risk_classifier.pkl, recovery_regressor.pkl, scaler.pkl

Output Files
Upon successful execution, the following files will be generated in the project directory:

output_chart.png: A visual representation of pain, swelling, and activity levels over the estimated recovery time.
<img width="4176" height="2074" alt="output_chart" src="https://github.com/user-attachments/assets/c4f5a79e-a96a-43a8-8f17-1cbcf3636cd8" />

intake_impact.png: A chart illustrating the impact of supplement intake on risk and recovery.
<img width="3569" height="1765" alt="intake_impact" src="https://github.com/user-attachments/assets/a29eddc7-995b-47f2-ad4d-927ecdbbb7f5" />

results_summary.txt: A text file summarizing all the predictions and comparisons, including a narrative summary.

risk_classifier.pkl: The saved trained Random Forest Classifier model.

recovery_regressor.pkl: The saved trained Gradient Boosting Regressor model.

scaler.pkl: The saved data scaler used for preprocessing.

Project Structure
.
├── varicose_vein_model.py  # The main script for the ML model
├── output_chart.png        # Generated recovery chart
├── intake_impact.png       # Generated intake impact chart
├── results_summary.txt     # Generated summary of results
├── risk_classifier.pkl     # Saved trained risk classification model
├── recovery_regressor.pkl  # Saved trained recovery prediction model
└── scaler.pkl              # Saved data scaler

Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please feel free to open an issue or submit a pull request.

Author

Aman Sharma
