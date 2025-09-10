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
Steps to set up and run the project.

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

**Output**

Recovery chart saved as 'output_chart.png', 
Intake impact chart saved as 'intake_impact.png', 
Results summary saved as 'results_summary.txt'


Output Files
After successful execution, the following files will be generated in the project directory:

output_chart.png: A visual representation of pain, swelling, and activity levels over the estimated recovery time.
<img width="4176" height="2074" alt="output_chart" src="https://github.com/user-attachments/assets/c4f5a79e-a96a-43a8-8f17-1cbcf3636cd8" />

intake_impact.png: A chart illustrating the impact of supplement intake on risk and recovery.
<img width="3569" height="1765" alt="intake_impact" src="https://github.com/user-attachments/assets/a29eddc7-995b-47f2-ad4d-927ecdbbb7f5" />

results_summary.txt: A text file summarizing all the predictions and comparisons, including a narrative summary.

risk_classifier.pkl: The saved trained Random Forest Classifier model.

recovery_regressor.pkl: The saved trained Gradient Boosting Regressor model.

scaler.pkl: The saved data scaler used for preprocessing.

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to open an issue or submit a pull request.

Author Aman Sharma LinkedIn :- www.linkedin.com/in/aman-sharma-842b66318
