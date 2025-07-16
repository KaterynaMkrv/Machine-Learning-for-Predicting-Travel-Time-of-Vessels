import pandas as pd
import pytest

from scripts.linear_regression import run_linear_regression
from scripts.random_forest import run_random_forest
from scripts.mlp_regressor_with_cv import run_mlp_regressor
from scripts.gradient_boosting_with_cv import run_gradient_boosting

# This test suite verifies that the four regression model functions:
# 1. Run successfully on a small sample dataset.
# 2. Return predictions as a pandas DataFrame containing "Actual" and "Predicted" columns.
# 3. Return evaluation metrics as a dictionary containing keys "MAE", "RMSE", and "R2".
# It ensures the models produce outputs in the expected format and include basic evaluation metrics.


# Create dummy test data
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "shiptype": [70, 70, 60],
        "Length": [150, 130, 120],
        "Draught": [6.5, 6.8, 6.1],
        "Distance_km": [200, 180, 220],
        "SOG_kmh": [25, 23, 26],
        "Trip_Duration_Hours": [8, 7.5, 9]
    })

# Basic functional test
@pytest.mark.parametrize("model_func", [
    run_linear_regression,
    run_random_forest,
    run_mlp_regressor,
    run_gradient_boosting
])
def test_model_runs(sample_df, model_func):
    preds, metrics = model_func(sample_df)
    assert isinstance(preds, pd.DataFrame), "Prediction output is not a DataFrame"
    assert "Actual" in preds.columns and "Predicted" in preds.columns, "Missing prediction columns"
    assert isinstance(metrics, dict), "Metrics not returned as a dictionary"
    assert "MAE" in metrics and "RMSE" in metrics and "R2" in metrics, "Missing evaluation metrics"
