
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scripts.geospatial_utils import add_predicted_coords

def run_linear_regression(df_cleaned):
    df = df_cleaned.copy()
    df["index"] = df.index

    df["time"] = pd.to_datetime(df["time"])
    df["EndTime"] = pd.to_datetime(df["EndTime"])
    df["ETA_hours"] = (df["EndTime"] - df["time"]).dt.total_seconds() / 3600

    features = ["shiptype", "Length", "Draught", "SOG_kmh", "COG", "index"]
    X = df[features]
    y = df["ETA_hours"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train.drop(columns=["index"]), y_train)
    y_pred = model.predict(X_test.drop(columns=["index"]))

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    preds_df = X_test.copy()
    preds_df["Actual"] = y_test.values
    preds_df["Predicted"] = y_pred
    preds_df = add_predicted_coords(preds_df, df)

    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 2)}
    return preds_df, metrics
