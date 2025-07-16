from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from scripts.geospatial_utils import add_predicted_coords
from scripts.prediction_utils import select_best_predictions



def run_gradient_boosting_with_cv(df):
    df = df.copy()  # ✅ Safely copy so you can add 'index' without side effects
    df["index"] = df.index  # ✅ Add index as column for later use

    # Features and target
    X = df[["shiptype", "Length", "Draught", "Distance_km", "SOG_kmh", "index"]]
    y = df["ETA_hours"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Model and param grid
    gbr = GradientBoostingRegressor(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0],
        'min_samples_split': [2, 5],
    }

    # Random search
    random_search = RandomizedSearchCV(
        gbr,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Prepare output
    preds_df = X_test.copy()
    preds_df["Actual"] = y_test
    preds_df["Predicted"] = y_pred

    preds_df = add_predicted_coords(preds_df, df)


    metrics = {
        "Best Parameters": random_search.best_params_,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return preds_df, metrics
