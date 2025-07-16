
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
from scripts.geospatial_utils import add_predicted_coords

def run_mlp_regressor_with_cv(df):
    df = df.copy()
    df["index"] = df.index

    # Add ETA target
    df["time"] = pd.to_datetime(df["time"])
    df["EndTime"] = pd.to_datetime(df["EndTime"])
    df["ETA_hours"] = (df["EndTime"] - df["time"]).dt.total_seconds() / 3600

    features = ["shiptype", "Length", "Draught", "SOG_kmh", "COG", "index"]
    X = df[features]
    y = df["ETA_hours"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [300]
    }

    mlp = MLPRegressor(random_state=42, early_stopping=True)

    grid_search = GridSearchCV(
        mlp, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train.drop(columns=["index"]), y_train)

    best_mlp = grid_search.best_estimator_
    y_pred = best_mlp.predict(X_test.drop(columns=["index"]))

    # Save best model
    dump(best_mlp, "cache/mlp_best_model.joblib")

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    preds_df = X_test.copy()
    preds_df["Actual"] = y_test.values
    preds_df["Predicted"] = y_pred

    preds_df = add_predicted_coords(preds_df, df)

    metrics = {
        "Best Parameters": grid_search.best_params_,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 2)
    }

    return preds_df, metrics
