import os
import io
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from models.predict import predictor
from utils.scenarios import get_scenarios
from utils.report_generator import generate_pdf_report

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Varicose Vein Recovery API")

# Add CORS middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
@app.get("/warmup")
def warmup():
    """Warm up the models to prevent cold start."""
    try:
        # Access the models to trigger loading if lazy
        status = predictor.risk_clf is not None and predictor.recovery_reg is not None
        return {"status": "warmed_up", "success": status}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class PredictionInput(BaseModel):
    age: int
    gender: str
    bmi: float
    pain_level: int = Field(..., ge=1, le=10)
    swelling_level: int = Field(..., ge=1, le=10)
    activity_level: int = Field(..., ge=1, le=10)
    beetroot_intake: str # None/Low/Medium/High
    fenugreek_intake: str # None/Low/Medium/High
    duration_weeks: int

class ReportInput(BaseModel):
    prediction_result: Dict[str, Any]
    input_data: Dict[str, Any]

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": predictor.risk_clf is not None}

@app.get("/model-info")
def model_info():
    # In a real scenario, we'd store these metrics during training
    return {
        "random_forest_accuracy": 0.943, # Hardcoded from original request or extracted from training logs if available
        "random_forest_auc_roc": 0.986,
        "training_samples": 50000,
        "model_version": "2.0.0",
        "features": [
            "age", "bmi", "gender_as_proxy", "pain_level", "swelling", 
            "activity_level", "beetroot_days", "beetroot_grams", 
            "fenugreek_days", "fenugreek_grams"
        ]
    }

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        data_dict = input_data.model_dump()
        result = predictor.predict_single(data_dict)
        scenarios = get_scenarios(data_dict)
        result["scenarios"] = scenarios
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        results = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            # Basic validation/defaulting for batch
            res = predictor.predict_single(row_dict)
            scenarios = get_scenarios(row_dict)
            res["scenarios"] = scenarios
            results.append(res)
        
        # Create output CSV
        output_df = df.copy()
        output_df['risk_level'] = [r['risk_level'] for r in results]
        output_df['recovery_weeks_mean'] = [r['recovery_weeks_mean'] for r in results]
        
        output_buffer = io.StringIO()
        output_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)
        
        # Return both JSON and the CSV (as a header or just JSON for now as per instructions)
        # The user asked for: "Returns predictions for all rows as JSON array. Also returns downloadable CSV with predictions appended."
        # FastAPI can't easily return two distinct response types in one go. 
        # Usually, this is handled by returning JSON and providing a link, or returning a multipart response.
        # I'll return the JSON, but I'll add a Base64 encoded CSV or just return the CSV if preferred.
        # Let's return JSON as the main response, and the user can request the CSV separately or we can provide a download link.
        # Actually, let's return the JSON array, and if they want the CSV they can use another endpoint or I can provide it as a base64 string.
        # Wait, "Returns predictions for all rows as JSON array. ALSO returns downloadable CSV".
        # I'll return a dictionary containing both.
        
        return {
            "predictions": results,
            "csv_content": output_buffer.getvalue()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-report")
def generate_report(input_data: ReportInput):
    try:
        pdf_buffer = generate_pdf_report(input_data.prediction_result, input_data.input_data)
        return StreamingResponse(
            pdf_buffer, 
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=recovery_report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
