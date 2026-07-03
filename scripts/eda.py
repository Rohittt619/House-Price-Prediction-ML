import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
os.makedirs("outputs/figures", exist_ok=True)

# Load dataset
df = pd.read_csv("data/processed/housing_cleaned.csv")

print("=" * 50)
print(df.head())

# -------------------------------
# Dataset Information
# -------------------------------
print("\nDataset Shape:", df.shape)
print("\nSummary Statistics:")
print(df.describe())

# -------------------------------
# Correlation Heatmap
# -------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/figures/correlation_heatmap.png")
plt.close()

# -------------------------------
# House Price Distribution
# -------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["MedHouseVal"], bins=30, kde=True)
plt.title("House Price Distribution")
plt.xlabel("Median House Value")
plt.tight_layout()
plt.savefig("outputs/figures/price_distribution.png")
plt.close()

# -------------------------------
# Median Income vs House Price
# -------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="MedInc",
    y="MedHouseVal",
    data=df,
    alpha=0.4
)
plt.title("Median Income vs House Price")
plt.tight_layout()
plt.savefig("outputs/figures/income_vs_price.png")
plt.close()

# -------------------------------
# House Age Distribution
# -------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["HouseAge"], bins=25)
plt.title("House Age Distribution")
plt.tight_layout()
plt.savefig("outputs/figures/house_age.png")
plt.close()

# -------------------------------
# Average Rooms vs House Price
# -------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="AveRooms",
    y="MedHouseVal",
    data=df,
    alpha=0.4
)
plt.title("Average Rooms vs House Price")
plt.tight_layout()
plt.savefig("outputs/figures/rooms_vs_price.png")
plt.close()

print("\nEDA Completed Successfully!")
print("Graphs saved inside outputs/figures/")