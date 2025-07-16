import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/k.jhnsn/Desktop/group04/data/Bremerhaven_Hamburg.csv")
df_cleaned = pd.read_csv("/Users/k.jhnsn/Desktop/group04/data/cleaned_Bremerhaven_Hamburg.csv")
# Select numeric features only
numeric_cols = df[[
    "Length", "Breadth", "Draught", "SOG", "COG", "TH"
]]
numeric_cols_cleaned = df[[
    "Length", "Breadth", "Draught", "SOG_km", "COG", "TH", "Distance_km", "ETT_minutes", "ETT_simulated_hr"
]]

# Compute correlation matrix
corr = numeric_cols.corr()
corr_cleaned = numeric_cols_cleaned.corr() 

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
sns.heatmap(corr_cleaned, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Numerical Features")
plt.show()