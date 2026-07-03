import streamlit as st
import pandas as pd
import joblib
import time

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = joblib.load("models/best_model.pkl")

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("🏠 Housing ML")

st.sidebar.markdown("### 📌 Project Information")

st.sidebar.write("**Dataset**")
st.sidebar.write("California Housing")

st.sidebar.write("**Algorithm**")
st.sidebar.write("Gradient Boosting Regressor")

st.sidebar.write("**Framework**")
st.sidebar.write("Scikit-Learn")

st.sidebar.divider()

st.sidebar.success("Developed by Rohit Rathod")

# -----------------------------------
# TITLE
# -----------------------------------

st.title("🏠 Housing Price Prediction")

st.write(
    "Predict California housing prices using Machine Learning."
)

st.divider()

# -----------------------------------
# INPUT SECTION
# -----------------------------------

st.subheader("🏡 Property Details")

col1, col2 = st.columns(2)

with col1:

    MedInc = st.slider(
        "Median Income",
        0.0,
        20.0,
        5.0
    )

    HouseAge = st.slider(
        "House Age",
        1,
        60,
        25
    )

    AveRooms = st.slider(
        "Average Rooms",
        1.0,
        15.0,
        5.5
    )

    AveBedrms = st.slider(
        "Average Bedrooms",
        0.5,
        5.0,
        1.0
    )

with col2:

    Population = st.slider(
        "Population",
        1,
        40000,
        1500
    )

    AveOccup = st.slider(
        "Average Occupancy",
        1.0,
        10.0,
        3.0
    )

    Latitude = st.slider(
        "Latitude",
        32.0,
        42.0,
        36.0
    )

    Longitude = st.slider(
        "Longitude",
        -125.0,
        -114.0,
        -120.0
    )

st.divider()

# -----------------------------------
# PREDICT BUTTON
# -----------------------------------

if st.button("🚀 Predict House Price", use_container_width=True):

    with st.spinner("Predicting..."):

        time.sleep(1)

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

    st.success("Prediction Completed Successfully!")

    st.metric(
        "💰 Estimated House Price",
        f"${prediction[0]*100000:,.0f}"
    )

st.divider()

# -----------------------------------
# MODEL PERFORMANCE
# -----------------------------------

st.subheader("📈 Model Performance")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("R² Score", "0.84")

with m2:
    st.metric("MAE", "0.32")

with m3:
    st.metric("RMSE", "0.46")

st.divider()

# -----------------------------------
# DATASET PREVIEW
# -----------------------------------

st.subheader("📄 Dataset Preview")

df = pd.read_csv("data/raw/housing.csv")

st.dataframe(
    df.head(),
    use_container_width=True
)

st.caption("Showing first 5 rows of the California Housing dataset.")