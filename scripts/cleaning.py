
import pandas as pd
import numpy as np
from scripts.geospatial_utils import calculate_bearing

REQUIRED_COLUMNS = ['TripID', 'StartTime', 'EndTime', 'time', 'Latitude', 'Longitude',
                    'Draught', 'Breadth', 'Length', 'SOG', 'COG', 'shiptype']

def clean_raw_data(path_or_df):
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df, on_bad_lines='skip')
    else:
        df = path_or_df.copy()

#Required columns check
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.drop(columns=[
        'Name', 'Callsign', 'AisSourcen', 'ID',
        'StartPort', 'EndPort', 'Destination'
    ], errors='ignore')

#datetimes parsing + check validity of row entries
    df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
    df['EndTime'] = pd.to_datetime(df['EndTime'], errors='coerce')
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    

    if df[['StartTime', 'EndTime', 'time']].isna().any().any():
        raise ValueError("Some datetime columns could not be parsed.")
    if not (df['EndTime'] >= df['StartTime']).all():
        raise ValueError("EndTime is earlier than StartTime in some rows.")
    
    df = df.sort_values(by=['TripID', 'time'])
#END

# # Convert numerics and validate bounds LAt, LON
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["SOG"] = pd.to_numeric(df["SOG"], errors="coerce")
    df["COG"] = pd.to_numeric(df["COG"], errors="coerce")
    df["Draught"] = pd.to_numeric(df["Draught"], errors="coerce")
    df["Breadth"] = pd.to_numeric(df["Breadth"], errors="coerce")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")

    # Drop rows with invalid GPS - removed, bcs very Hamburg specific. Any file can be uploaded
    """ df = df[(df['Latitude'] >= 52.0) & (df['Latitude'] <= 55.0)]
    df = df[(df['Longitude'] >= 6.0) & (df['Longitude'] <= 11.5)] """

    
    # Fill missing dimension values by shiptype median
    for col in ['Draught', 'Breadth', 'Length']:
        if df[col].isna().any():
            df[col] = df.groupby('shiptype')[col].transform(lambda x: x.fillna(x.median()))

    for col, max_val in [('Length', 400), ('Breadth', 60), ('Draught', 20)]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df[col] <= max_val]

    # Check for excessive missing values
    if df['SOG'].isna().mean() > 0.5 or df['COG'].isna().mean() > 0.5:
        raise ValueError("Too many missing values in SOG or COG columns.")
    
    # Filter SOG in km/h
    df['SOG_kmh'] = df['SOG'] * 1.852
    df = df[(df['SOG_kmh'] > 0) & (df['SOG_kmh'] < 70)]

    df['Distance_km'] = np.nan
    df['Bearing'] = np.nan

    for trip_id, group in df.groupby('TripID'):
        try:
            start = group.iloc[0]
            end = group.iloc[-1]
            dist = np.sqrt((end['Latitude'] - start['Latitude'])**2 + (end['Longitude'] - start['Longitude'])**2) * 111
            bear = calculate_bearing(start['Latitude'], start['Longitude'], end['Latitude'], end['Longitude'])
            df.loc[group.index, 'Distance_km'] = dist
            df.loc[group.index, 'Bearing'] = bear
        except:
            continue

    df['Trip_Duration_Hours'] = (df['EndTime'] - df['StartTime']).dt.total_seconds() / 3600
    return df
