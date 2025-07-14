Varicose Vein Recovery ML Model

This project implements a machine learning model to simulate and demonstrate the potential health benefits of a multivitamin syrup, specifically formulated with beetroot and fenugreek, in the treatment and recovery of varicose veins. This model uses synthetic data to illustrate the hypothesized efficacy of these natural ingredients.

Project Goal
The primary objective of this project is to provide a computational simulation that illustrates how a combined multivitamin syrup of beetroot and fenugreek can lead to desired improvements in varicose vein symptoms and recovery time. The output is designed to be clear and informative, showcasing the model's capabilities.

Features
Terminal-Based Input: Users can provide patient-specific data directly through the command line.

Dual Machine Learning Models:

RandomForestClassifier: Classifies the patient's varicose vein risk level (Low, Moderate, High).

GradientBoostingRegressor: Estimates the recovery time in weeks.

Synthetic Data Generation: Creates a realistic dataset (5000-10000 samples) for model training and validation.

Chart Generation:

Generates a output_chart.png visualizing the decrease in pain and swelling, and an increase in activity level over the estimated recovery period, on a dark background.

Generates an intake_impact.png showing the correlation between supplement intake and recovery metrics.

Comprehensive Output: Provides detailed predictions, including risk level, estimated recovery time, and a crucial "Supplement Comparison" section that highlights the impact of beetroot and fenugreek intake versus no supplements or optimal intake.

Results Summary: Saves a results_summary.txt file containing all prediction details and a narrative summary.

Model Persistence: Saves the trained models (.pkl files) for future use without retraining.

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
Follow these steps to set up and run the project on your local machine.

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
