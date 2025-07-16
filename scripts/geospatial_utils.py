from geopy.distance import distance
from geopy import Point
import numpy as np


def safe_parse_latlon(coord_str):
    """Safely parses a malformed coordinate string with multiple dots"""
    coord_str = str(coord_str).strip()

    if coord_str.count('.') > 1:
        parts = coord_str.split('.')
        cleaned = parts[0] + '.' + ''.join(parts[1:2])  # keep only first dot and one chunk after
    else:
        cleaned = coord_str

    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def calculate_bearing(start_lat, start_lon, end_lat, end_lon):
    start_lat = np.radians(start_lat)
    start_lon = np.radians(start_lon)
    end_lat = np.radians(end_lat)
    end_lon = np.radians(end_lon)

    d_lon = end_lon - start_lon

    x = np.sin(d_lon) * np.cos(end_lat)
    y = np.cos(start_lat) * np.sin(end_lat) - np.sin(start_lat) * np.cos(end_lat) * np.cos(d_lon)

    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

def estimate_destination(lat, lon, bearing, distance_km):
    start_point = Point(lat, lon)
    destination = distance(kilometers=distance_km).destination(start_point, bearing)
    return destination.latitude, destination.longitude


""" OLD: def add_predicted_coords(preds_df, original_df):
    pred_lats = []
    pred_lons = []

    for _, row in preds_df.iterrows():
        orig_idx = row['index']  # старый индекс исходного датафрейма
        start_lat = original_df.loc[orig_idx, 'Latitude']
        start_lon = original_df.loc[orig_idx, 'Longitude']
        bearing = original_df.loc[orig_idx, 'COG']
        sog = original_df.loc[orig_idx, 'SOG_kmh']

        pred_dist_km = row['Predicted'] * sog
        pred_lat, pred_lon = estimate_destination(start_lat, start_lon, bearing, pred_dist_km)

        pred_lats.append(pred_lat)
        pred_lons.append(pred_lon)

    preds_df['Pred_Lat'] = pred_lats
    preds_df['Pred_Lon'] = pred_lons
    preds_df['Actual_Lat'] = preds_df['index'].map(original_df['Latitude'])
    preds_df['Actual_Lon'] = preds_df['index'].map(original_df['Longitude'])
    preds_df['TripID'] = preds_df['index'].map(original_df['TripID'])
    preds_df.to_csv("/Users/k.jhnsn/Desktop/group04/data/predicted data.csv", index=False)

    return preds_df """

def add_predicted_coords(preds_df, original_df):
    import pandas as pd

    pred_lats = []
    pred_lons = []

    original_df['time'] = pd.to_datetime(original_df['time'])

    for _, row in preds_df.iterrows():
        trip_id = original_df.loc[row['index'], 'TripID']
        trip_df = original_df[original_df['TripID'] == trip_id].sort_values("time")

        start_time = trip_df.iloc[0]['time']
        predicted_hours = row['Predicted']
        predicted_time = start_time + pd.to_timedelta(predicted_hours, unit='h')

        # Find the closest point to predicted arrival time
        closest_row = trip_df.iloc[(trip_df['time'] - predicted_time).abs().argsort()[:1]]

        pred_lats.append(closest_row.iloc[0]['Latitude'])
        pred_lons.append(closest_row.iloc[0]['Longitude'])

    preds_df['Pred_Lat'] = pred_lats
    preds_df['Pred_Lon'] = pred_lons
    preds_df['Actual_Lat'] = preds_df['index'].map(original_df['Latitude'])
    preds_df['Actual_Lon'] = preds_df['index'].map(original_df['Longitude'])
    preds_df['TripID'] = preds_df['index'].map(original_df['TripID'])
    predicted_times = []

    # Add Predicted arrival time if time info is available
    if "TripID" in preds_df.columns and "time" in original_df.columns:
        predicted_times = []

        original_df["time"] = pd.to_datetime(original_df["time"])

        for idx, row in preds_df.iterrows():
            trip_id = row["TripID"]
            predicted_hours = row["Predicted"]

            trip_df = original_df[original_df["TripID"] == trip_id]
            if trip_df.empty:
                predicted_times.append(pd.NaT)
                continue

            start_time = trip_df["time"].min()
            predicted_time = start_time + pd.to_timedelta(predicted_hours, unit="h")
            predicted_times.append(predicted_time)

            if "Actual" in preds_df.columns:
                preds_df["Error_hr"] = preds_df["Predicted"] - preds_df["Actual"]


        preds_df["Predicted arrival time"] = predicted_times
        preds_df.to_csv("data/predicted_data.csv", index=False)

    return preds_df
