import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/best_model.pkl")

print("=" * 50)
print("Housing Price Prediction")
print("=" * 50)

# Take user input
MedInc = float(input("Median Income: "))
HouseAge = float(input("House Age: "))
AveRooms = float(input("Average Rooms: "))
AveBedrms = float(input("Average Bedrooms: "))
Population = float(input("Population: "))
AveOccup = float(input("Average Occupancy: "))
Latitude = float(input("Latitude: "))
Longitude = float(input("Longitude: "))

sample = pd.DataFrame({
    "MedInc":[MedInc],
    "HouseAge":[HouseAge],
    "AveRooms":[AveRooms],
    "AveBedrms":[AveBedrms],
    "Population":[Population],
    "AveOccup":[AveOccup],
    "Latitude":[Latitude],
    "Longitude":[Longitude]
})

prediction = model.predict(sample)

print("\nPredicted House Price:")
print(f"${prediction[0]*100000:,.0f}")