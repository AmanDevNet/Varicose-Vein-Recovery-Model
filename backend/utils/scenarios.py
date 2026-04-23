from typing import Dict, Any
from models.predict import predictor

def get_scenarios(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Compare three scenarios:
    1. no_supplements: beetroot and fenugreek = None
    2. current_intake: input as provided
    3. optimal_intake: beetroot and fenugreek = High
    """
    
    # 1. No Supplements
    data_no = data.copy()
    data_no["beetroot_intake"] = "None"
    data_no["fenugreek_intake"] = "None"
    res_no = predictor.predict_single(data_no)
    
    # 2. Current Intake
    res_curr = predictor.predict_single(data)
    
    # 3. Optimal Intake
    data_opt = data.copy()
    data_opt["beetroot_intake"] = "High"
    data_opt["fenugreek_intake"] = "High"
    res_opt = predictor.predict_single(data_opt)
    
    return {
        "no_supplements": {
            "risk": res_no["risk_level"],
            "recovery_weeks": res_no["recovery_weeks_mean"]
        },
        "current_intake": {
            "risk": res_curr["risk_level"],
            "recovery_weeks": res_curr["recovery_weeks_mean"]
        },
        "optimal_intake": {
            "risk": res_opt["risk_level"],
            "recovery_weeks": res_opt["recovery_weeks_mean"]
        }
    }
