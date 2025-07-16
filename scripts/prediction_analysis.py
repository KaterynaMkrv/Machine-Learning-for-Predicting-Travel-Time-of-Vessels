import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def analyze_predictions(preds_df: pd.DataFrame, save_plots: bool = False, output_dir: str = "analysis_outputs"):
    preds_df["Error_hr"] = preds_df["Predicted"] - preds_df["Actual"]
    preds_df["Abs_Error_hr"] = preds_df["Error_hr"].abs()

    summary_stats = {
        "Mean Error (hr)": preds_df["Error_hr"].mean(),
        "Median Error (hr)": preds_df["Error_hr"].median(),
        "75th Percentile (hr)": preds_df["Error_hr"].quantile(0.75),
        "Max Error (hr)": preds_df["Error_hr"].max(),
        "Min Error (hr)": preds_df["Error_hr"].min()
    }

    summary_df = pd.DataFrame([summary_stats])

    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Error distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(preds_df["Error_hr"], bins=30, kde=True, color="skyblue")
        plt.title("Prediction Error Distribution")
        plt.xlabel("Error (Predicted - Actual) [hr]")
        plt.ylabel("Frequency")
        plt.tight_layout()

        if save_plots:
            plt.savefig(f"{output_dir}/error_distribution.png")
            plt.close()
        else:
            st.subheader("Prediction Error Distribution")
            st.pyplot(plt.gcf())  # ⬅️ Displays plot in Streamlit


        # Heatmap of absolute errors
        if "Pred_Lat" in preds_df.columns and "Pred_Lon" in preds_df.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=preds_df,
                x="Pred_Lon", y="Pred_Lat",
                hue="Abs_Error_hr", palette="coolwarm",
                size="Abs_Error_hr", sizes=(20, 200)
            )
            plt.title("Spatial Distribution of Absolute Errors")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.legend(title="Abs Error (hr)")
            plt.tight_layout()

            if save_plots:
                plt.savefig(f"{output_dir}/error_heatmap.png")
                plt.close()
            else:
                st.subheader("Spatial Error Heatmap")
                st.pyplot(plt.gcf())


        # Outlier detection: top 5 highest absolute errors

        if not save_plots:
            st.subheader("Top 5 Outlier Predictions by Absolute Error")
            st.dataframe(outliers_df)
        
        outliers_df = preds_df.nlargest(5, "Abs_Error_hr")
        outliers_df.to_csv(f"{output_dir}/top_5_outliers.csv", index=False)

    return summary_df, preds_df
