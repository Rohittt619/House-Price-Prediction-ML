import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/housing.csv")

print("=" * 50)
print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Save cleaned dataset
df.to_csv("data/processed/housing_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
print(df.shape)