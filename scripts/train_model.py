import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
df = pd.read_csv("data/processed/housing_cleaned.csv")

# Features & Target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("=" * 60)
print("Training Models...")
print("=" * 60)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        random_state=42,
        n_estimators=100
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

best_model = None
best_score = -999

os.makedirs("models", exist_ok=True)

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(
        y_test,
        pred,
        squared=False
    )
    r2 = r2_score(y_test, pred)

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    if r2 > best_score:
        best_score = r2
        best_model = model

joblib.dump(best_model, "models/best_model.pkl")

print("\n" + "=" * 60)
print("✅ Best model saved!")
print("Location : models/best_model.pkl")
print("=" * 60)