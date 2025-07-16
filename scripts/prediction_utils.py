from geopy.distance import geodesic
from scripts.geospatial_utils import add_predicted_coords

def calculate_deviation_km(row):
    actual = (row['Actual_Lat'], row['Actual_Lon'])
    predicted = (row['Pred_Lat'], row['Pred_Lon'])
    return geodesic(actual, predicted).km

def select_best_predictions(preds_df, top_n=1000):
    """
    Select a diverse sample of low-error predictions for mapping.

    - Sort by absolute error.
    - Take a wider top-k pool (e.g., 5x top_n).
    - Randomly sample from this pool to ensure diversity.
    """
    preds_df = preds_df.copy()
    preds_df["AbsError"] = (preds_df["Actual"] - preds_df["Predicted"]).abs()
    sorted_preds = preds_df.sort_values("AbsError")

    pool_size = min(top_n * 5, len(sorted_preds))
    diverse_sample = sorted_preds.head(pool_size).sample(n=min(top_n, pool_size), random_state=42)
    return diverse_sample


