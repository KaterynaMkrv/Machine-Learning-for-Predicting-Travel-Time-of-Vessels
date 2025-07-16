import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scripts.geospatial_utils import add_predicted_coords
from scripts.prediction_utils import select_best_predictions


def run_random_forest(df_cleaned):
    df = df_cleaned.copy()
    df["index"] = df.index  # track rows for matching later

    # Features for per-row ETA prediction
    features = ["shiptype", "Length", "Draught", "SOG_kmh", "COG"]
    X = df[features + ["index"]]
    y = df["ETA_hours"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train[features], y_train)

    # Predict
    y_pred = model.predict(X_test[features])

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Prepare output DataFrame
    preds_df = X_test.copy()
    preds_df["Actual"] = y_test.values
    preds_df["Predicted"] = y_pred

    # Use time-based coordinate matching logic
    preds_df = add_predicted_coords(preds_df, df)

    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 2)
    }

    return preds_df, metrics
