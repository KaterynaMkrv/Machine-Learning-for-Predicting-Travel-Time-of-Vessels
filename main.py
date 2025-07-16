import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os
from joblib import dump, load
import hashlib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.random_forest import run_random_forest
from scripts.random_forest_with_cv import run_random_forest_with_cv
from scripts.linear_regression import run_linear_regression
from scripts.ridge_regression import run_linear_regression_with_cv
from scripts.mlp_regressor import run_mlp_regressor
from scripts.mlp_regressor_with_cv import run_mlp_regressor_with_cv
from scripts.gradient_boosting import run_gradient_boosting
from scripts.gradient_boosting_with_cv import run_gradient_boosting_with_cv
from scripts.geospatial_utils import calculate_bearing, safe_parse_latlon
from scripts.cleaning import clean_raw_data
from scripts.broker_agent import broker_agent
from scripts.prediction_analysis import analyze_predictions

def get_file_hash(uploaded_file):
    if uploaded_file is None:
        return "default"
    uploaded_file.seek(0)
    content = uploaded_file.read()
    uploaded_file.seek(0)

    return hashlib.md5(content).hexdigest()

st.set_page_config(page_title="Ship Trip Duration Prediction", layout="wide")
st.title("🚢 Ship Trip Duration Prediction App")

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_Bremerhaven_Hamburg.csv")

uploaded_file = st.file_uploader("Upload your CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines="skip")
        df = clean_raw_data(df)
        df["time"] = pd.to_datetime(df["time"])
        df["EndTime"] = pd.to_datetime(df["EndTime"])
        if "ETA_hours" not in df.columns and "Trip_Duration_Hours" in df.columns:
            df.rename(columns={"Trip_Duration_Hours": "ETA_hours"}, inplace=True)
        elif "ETA_hours" not in df.columns:
            df["ETA_hours"] = (df["EndTime"] - df["time"]).dt.total_seconds() / 3600
        st.success("File uploaded and cleaned successfully.")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        st.stop()

else: #if NONE, choose default
    df = load_data()
    df["time"] = pd.to_datetime(df["time"])
    df["EndTime"] = pd.to_datetime(df["EndTime"])
    df["ETA_hours"] = (df["EndTime"] - df["time"]).dt.total_seconds() / 3600

df["ETA_hours"] = (df["EndTime"] - df["time"]).dt.total_seconds() / 3600
st.write("### Sample Data", df.head())

prediction_mode = st.radio("Choose prediction mode", ["Individual Model", "Broker Agent"], key="mode_selector")

#---------------Individual START 
if prediction_mode == "Individual Model":
    model_options = {
        "Random Forest": (run_random_forest, run_random_forest_with_cv),
        "Linear Regression": (run_linear_regression, run_linear_regression_with_cv),
        "MLP Regressor": (run_mlp_regressor, run_mlp_regressor_with_cv),
        "Gradient Boosting": (run_gradient_boosting, run_gradient_boosting_with_cv),
    }
    model_name = st.selectbox("Choose a model:", list(model_options.keys()), key="model_selector")
    version = st.radio("Model version:", ["Initial Model", "Tuned Model"], key="version_selector")
    force_retrain = st.checkbox("Force model re-run", key="retrain_checkbox")
    
    if st.button("Run Individual Model"):
        with st.spinner("Running model..."):
            
            file_hash = get_file_hash(uploaded_file)
            
            if uploaded_file is not None:
                key = f"{model_name}_{version.replace(' ', '_')}_{uploaded_file.name}"
                cache_path = f"cache/preds_{key}.joblib"
                
            else:
                key = f"{model_name}_{version.replace(' ', '_')}_default"
                cache_path = f"cache/preds_{key}.joblib"

            if os.path.exists(cache_path) and not force_retrain:
                loaded = load(cache_path)

                if isinstance(loaded, tuple) and len(loaded) == 2:
                    preds_df, metrics = loaded
                else:
                    preds_df = loaded
                    metrics = {"Info": "Loaded from cache (metrics not stored in old format)"}

                st.success("Loaded predictions from cache.")
            else:
                model_func = model_options[model_name][0] if version == "Initial Model" else model_options[model_name][1]
                preds_df, metrics = model_func(df)
                os.makedirs("cache", exist_ok=True)
                dump((preds_df, metrics), cache_path)  # Save as tuple
                st.success("Model run complete. Predictions cached.")
            

        st.session_state["preds_df"] = preds_df
        st.session_state["metrics"] = metrics
        st.session_state["model_name"] = model_name
        st.session_state["version"] = version
#---------------Individual END 

#------------BROKER AGENT START-----------------
elif prediction_mode == "Broker Agent":
    st.markdown("### Set model weights for Broker Agent")
    w_rf = st.slider("Random Forest weight", 0.0, 1.0, 0.3, key="w_rf")
    w_lrg = st.slider("Linear Regression weight", 0.0, 1.0, 0.3, key="w_lrg")
    w_mlp = st.slider("MLP weight", 0.0, 1.0, 0.2, key="w_mlp")
    w_gb = st.slider("Gradient Boosting weight", 0.0, 1.0, 0.2, key="w_gb")
    if st.button("Run Broker Agent"):
        total = w_rf + w_lrg + w_mlp + w_gb
        if total == 0:
            st.warning("Total weight must be > 0.")
            st.stop()


        weights = [w_rf / total, w_mlp / total, w_gb / total, w_lrg / total]
        
        if uploaded_file is not None:
            file_hash = uploaded_file.name
        else:
            file_hash = "default"

        preds_df, metrics = broker_agent(df, weights=weights, file_hash=file_hash)
        st.write("Cache key:", file_hash)


        st.session_state["preds_df"] = preds_df
        st.session_state["metrics"] = metrics
        st.session_state["model_name"] = "Broker Agent"
        st.session_state["version"] = "Weighted Ensemble"
#------------BROKER AGENT END -----------------


#-------- In all scenarios show graph and a map and location based map

if "preds_df" in st.session_state and "metrics" in st.session_state: 
    preds_df = st.session_state["preds_df"]
    st.write(f"### Metrics for {st.session_state['model_name']} ({st.session_state['version']})")
    st.write(st.session_state["metrics"])
    if "Actual" in preds_df.columns and "Predicted" in preds_df.columns:
        #fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(5, 2.5))  # Width=5 inches, Height=4 inches
        sns.scatterplot(data=preds_df, x="Actual", y="Predicted", ax=ax, s = 10)   #GRAPH
        ax.plot([preds_df["Actual"].min(), preds_df["Actual"].max()],
                [preds_df["Actual"].min(), preds_df["Actual"].max()], 'r--')
        ax.set_title("Actual vs Predicted Trip Duration")
        st.pyplot(fig)


        with st.expander("Predict Arrival Time based on Location"):  # MAP based on LOCATION

            st.markdown("Location Input")
            location_file = st.file_uploader("Upload ship location file (Lat/Lon)", type=["csv"])

            if location_file:
                try:
                    user_locs = pd.read_csv(location_file)

                    if not {"Latitude", "Longitude"}.issubset(user_locs.columns):
                        raise ValueError("Uploaded file must contain 'Latitude' and 'Longitude' columns.")
                    #st.write("Data uploaded", user_locs.head())

                    results = []
                    for _, point in user_locs.iterrows():
                        lat = safe_parse_latlon(point["Latitude"])
                        lon = safe_parse_latlon(point["Longitude"])

                        if pd.isna(lat) or pd.isna(lon):
                            st.warning(f"Invalid coordinate skipped: {point['Latitude']}, {point['Longitude']}")
                            continue

                        exact_match = preds_df[(preds_df["Pred_Lat"] == lat) & (preds_df["Pred_Lon"] == lon)]
                        if not exact_match.empty:
                            row = exact_match.iloc[0]
                        else:
                            distances = np.sqrt((preds_df["Pred_Lat"] - lat)**2 + (preds_df["Pred_Lon"] - lon)**2)
                            nearest_idx = distances.idxmin()
                            row = preds_df.loc[nearest_idx]
                        #st.write("Distances calculated", distances)

                        results.append({
                            "Latitude": lat,
                            "Longitude": lon,
                            "ETT_estimated_hr": row.get("Predicted", np.nan),
                            "Predicted arrival time": row.get("Predicted arrival time", "N/A"),
                            "Error_hr": row.get("Error_hr", "N/A")
                        })

                    
                    results_df = pd.DataFrame(results)

                    if results_df.empty:
                        st.warning("No matching predictions found for provided coordinates.")
                        st.stop()


                    if st.button("Generate Results on a Map"):
                        fig_map = go.Figure()
                        fig_map.add_trace(go.Scattermapbox(
                            lat=df["Latitude"], lon=df["Longitude"],
                            mode="markers", name="Historical",
                            marker=dict(size=4, color="blue")
                        ))
                        fig_map.add_trace(go.Scattermapbox(
                            lat=results_df["Latitude"], lon=results_df["Longitude"],
                            mode="markers", name="Provided Location Points",
                            marker=dict(size=5, color="red"),
                            text=[
                                f"ETA: {row['ETT_estimated_hr']:.2f} h<br> Arrival Time: {row['Predicted arrival time']}"
                                for _, row in results_df.iterrows()
                            ],
                            customdata=results_df[["ETT_estimated_hr", "Predicted arrival time"]],
                            hovertemplate="ETA: %{customdata[0]:.2f} h<br>Arrival: %{customdata[1]}<extra></extra>"                             
                        ))
                        
                        fig_map.update_layout(
                            mapbox=dict(style="open-street-map",
                                        center=dict(lat=df["Latitude"].mean(), lon=df["Longitude"].mean()),
                                        zoom=7),
                            hovermode="closest",
                            margin=dict(r=0, l=0, t=0, b=0),
                            height=600
                        )
                        
                        st.write("📍 Location Prediction Results Preview", results_df.head())

                        # Download button
                        csv = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Predictions for uploaded coordinates Results as CSV",
                            data=csv,
                            file_name="predicted_arrival_location_points_results.csv",
                            mime="text/csv"
                        )

                        st.plotly_chart(fig_map, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Could not process uploaded file: {e}")

        with st.expander("Show Historical/Predicted Arrival Map"): #MAP
            if st.button("Generate Map"):
                fig_map = go.Figure()
                fig_map.add_trace(go.Scattermapbox(
                    lat=df["Latitude"], lon=df["Longitude"],
                    mode="markers", name="Historical",
                    marker=dict(size=4, color="blue")
                ))
                fig_map.add_trace(go.Scattermapbox(
                    lat=preds_df["Pred_Lat"], lon=preds_df["Pred_Lon"],
                    mode="markers", name="Predicted Arrival",
                    marker=dict(size=3, color="red")
                ))
                fig_map.update_layout(
                    mapbox=dict(style="open-street-map",
                                center=dict(lat=df["Latitude"].mean(), lon=df["Longitude"].mean()),
                                zoom=7),
                    margin=dict(r=0, l=0, t=0, b=0),
                    height=600
                )
                # Download button
                csv = preds_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                            label="📥 Download Prediction Results as CSV",
                            data=csv,
                            file_name="predicted_arrival_results.csv",
                            mime="text/csv"
                        )
                st.plotly_chart(fig_map, use_container_width=True)
                st.write("📍 Prediction Results Preview", preds_df.head())

        with st.expander("Show Predicted Error Analysis"): #errors
            if "preds_df" in st.session_state:
                if st.button("Provide error Metrics"):

                    summary_df, preds_df = analyze_predictions(preds_df, save_plots=False)
                    st.subheader("Prediction Summary Statistics")
                    st.dataframe(summary_df.head())

                    st.subheader("Detailed Predictions with Error")
                    st.dataframe(preds_df.head())

                    # Download button
                    csv = preds_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                            label="📥 Download Predictions with Errors Results as CSV",
                            data=csv,
                            file_name="predicted_error_results.csv",
                            mime="text/csv"
                        )


                    # two columns for side-by-side plots 
                    col1, col2 = st.columns(2)

                    with col1: #error distribution
                        fig1, ax1 = plt.subplots(figsize=(5, 4))
                        sns.histplot(preds_df["Error_hr"], bins=30, kde=True, color="skyblue", ax=ax1)
                        ax1.set_title("Prediction Error Distribution")
                        ax1.set_xlabel("Error (Predicted - Actual) [hr]")
                        ax1.set_ylabel("Frequency")
                        st.pyplot(fig1)

                    with col2: #heatmap
                        if "Pred_Lat" in preds_df.columns and "Pred_Lon" in preds_df.columns:
                            fig2, ax2 = plt.subplots(figsize=(5, 4))
                            scatter = sns.scatterplot(
                                data=preds_df,
                                x="Pred_Lon",
                                y="Pred_Lat",
                                hue="Abs_Error_hr",
                                palette="coolwarm",
                                size="Abs_Error_hr",
                                sizes=(20, 200),
                                ax=ax2,
                                legend=False
                            )
                            ax2.set_title("Spatial Distribution of Absolute Errors")
                            ax2.set_xlabel("Longitude")
                            ax2.set_ylabel("Latitude")
                            st.pyplot(fig2)






