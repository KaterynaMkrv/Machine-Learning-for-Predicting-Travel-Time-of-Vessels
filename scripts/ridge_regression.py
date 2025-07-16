""" I used Ridge Regression to prevent the model from overfitting by keeping the coefficients small. 
The alpha parameter controls how strong this penalty is, 
and I tuned alpha to find the best balance between fitting well and keeping the model simple."""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scripts.geospatial_utils import add_predicted_coords
from scripts.prediction_utils import select_best_predictions


def run_linear_regression_with_cv(df):
    # Select features and target
    X = df[["shiptype", "Length", "Draught", "Distance_km", "SOG_kmh"]]
    y = df["Trip_Duration_Hours"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Ridge regression with hyperparameter tuning for alpha
    ridge = Ridge(random_state=42)
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

    grid_search = GridSearchCV(
        ridge,
        param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predict
    y_pred = best_model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Prepare output
    preds_df = X_test.copy()
    preds_df["Actual"] = y_test
    preds_df["Predicted"] = y_pred

    # Add geo predicted endpoint coordinates
    preds_df = add_predicted_coords(preds_df, df)


    metrics = {
        "Best Alpha": grid_search.best_params_['alpha'],
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return preds_df, metrics
