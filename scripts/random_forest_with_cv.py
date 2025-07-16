from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from scripts.geospatial_utils import add_predicted_coords
from scripts.prediction_utils import select_best_predictions


def run_random_forest_with_cv(df_cleaned):
    df = df_cleaned.copy()
    df["index"] = df.index

    # Ensure time columns are datetime
    df["time"] = pd.to_datetime(df["time"])
    df["EndTime"] = pd.to_datetime(df["EndTime"])
    df["ETA_hours"] = (df["EndTime"] - df["time"]).dt.total_seconds() / 3600

    # Encode shiptype if necessary
    if df["shiptype"].dtype == 'object':
        le = LabelEncoder()
        df["shiptype"] = le.fit_transform(df["shiptype"])

    # Feature selection
    features = ["shiptype", "Length", "Draught", "SOG_kmh", "COG", "index"]
    X = df[features]
    y = df["ETA_hours"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Randomized search parameters
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train.drop(columns=["index"]), y_train)

    best_rf = random_search.best_estimator_
    y_pred = best_rf.predict(X_test.drop(columns=["index"]))

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Create prediction DataFrame
    preds_df = X_test.copy()
    preds_df["Actual"] = y_test.values
    preds_df["Predicted"] = y_pred

    # Convert ETA into predicted position on the real route
    preds_df = add_predicted_coords(preds_df, df)

    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 2),
        "Best_Params": random_search.best_params_
    }

    return preds_df, metrics