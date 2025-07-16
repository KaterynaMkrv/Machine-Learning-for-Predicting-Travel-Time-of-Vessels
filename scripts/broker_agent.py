import os
import hashlib
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.random_forest import run_random_forest
from scripts.linear_regression import run_linear_regression
from scripts.mlp_regressor import run_mlp_regressor
from scripts.gradient_boosting import run_gradient_boosting
from scripts.geospatial_utils import add_predicted_coords

def get_file_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def broker_agent(df: pd.DataFrame, weights: list[float], file_hash: str = "default") -> tuple[pd.DataFrame, dict]:
    os.makedirs("cache", exist_ok=True)
    model_name = "Broker Agent"

    model_funcs = {
        "Random Forest": run_random_forest,
        "Linear Regression": run_linear_regression,
        "MLP Regressor": run_mlp_regressor,
        "Gradient Boosting": run_gradient_boosting
    }

    preds = {}
    for name, func in model_funcs.items():
        key = f"preds_{name}_Initial_Model_{file_hash}.joblib"
        cache_path = f"cache/{key}"

        if os.path.exists(cache_path):
            preds[name] = load(cache_path)
        else:
            pred_df, _ = func(df)
            dump(pred_df, cache_path)
            preds[name] = pred_df

    # Ensure alignment
    for key in preds:
        preds[key] = preds[key].reset_index(drop=True)

    actual = preds["Random Forest"]["Actual"]
    final_pred = (
        weights[0] * preds["Random Forest"]["Predicted"] +
        weights[1] * preds["Linear Regression"]["Predicted"] +
        weights[2] * preds["MLP Regressor"]["Predicted"] +
        weights[3] * preds["Gradient Boosting"]["Predicted"]
    )

    result_df = preds["Gradient Boosting"].copy()
    result_df["Predicted"] = final_pred
    result_df = add_predicted_coords(result_df, df)

    metrics = {
        "MAE": round(mean_absolute_error(actual, final_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(actual, final_pred)), 2),
        "R2": round(r2_score(actual, final_pred), 2)
    }

    return result_df, metrics
