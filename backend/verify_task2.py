import requests
import json
import pandas as pd
import io

BASE_URL = "http://127.0.0.1:8000"

def test_endpoints():
    # 1. Health
    print("Testing /health...")
    res = requests.get(f"{BASE_URL}/health")
    print(res.json())
    
    # 2. Model Info
    print("\nTesting /model-info...")
    res = requests.get(f"{BASE_URL}/model-info")
    print(res.json())
    
    # 3. Predict
    print("\nTesting /predict...")
    payload = {
        "age": 45,
        "gender": "Female",
        "bmi": 26.5,
        "pain_level": 6,
        "swelling_level": 4,
        "activity_level": 5,
        "beetroot_intake": "Medium",
        "fenugreek_intake": "Low",
        "duration_weeks": 4
    }
    res = requests.post(f"{BASE_URL}/predict", json=payload)
    prediction = res.json()
    print(json.dumps(prediction, indent=2))
    
    # 4. Generate Report
    print("\nTesting /generate-report...")
    report_payload = {
        "prediction_result": prediction,
        "input_data": payload
    }
    res = requests.post(f"{BASE_URL}/generate-report", json=report_payload)
    if res.status_code == 200:
        with open("test_report.pdf", "wb") as f:
            f.write(res.content)
        print("Report saved as test_report.pdf")
    else:
        print(f"Report generation failed: {res.text}")

    # 5. Batch Predict
    print("\nTesting /batch-predict...")
    df = pd.DataFrame([payload, payload]) # Two identical rows
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    files = {'file': ('test.csv', csv_buffer.getvalue(), 'text/csv')}
    res = requests.post(f"{BASE_URL}/batch-predict", files=files)
    print("Batch predict response received.")
    batch_res = res.json()
    print(f"Number of predictions: {len(batch_res['predictions'])}")
    # print(f"CSV Content preview: {batch_res['csv_content'][:100]}...")

if __name__ == "__main__":
    try:
        test_endpoints()
    except Exception as e:
        print(f"Test failed: {e}")
