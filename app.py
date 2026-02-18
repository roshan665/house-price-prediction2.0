import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# -------------------------
# Load Dataset
# -------------------------
data = pd.read_csv("cleaned_real_estate_dataset.csv")

X = data.drop("price", axis=1)
y = data["price"]

# Define columns
categorical_cols = ["location"]
numeric_cols = ["bhk", "bathroom", "balcony", "area_sqft", "price_per_sqft"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

# Train model
model.fit(X, y)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🏠 House Price Prediction App")

st.write("Enter house details below:")

location = st.selectbox("Location", data["location"].unique())
bhk = st.slider("BHK", 1, 5, 2)
bathroom = st.slider("Bathrooms", 1, 4, 2)
balcony = st.slider("Balconies", 0, 3, 1)
area_sqft = st.number_input("Area (sqft)", min_value=500, max_value=3000, value=1000)
price_per_sqft = st.number_input("Price per sqft", min_value=4000, max_value=15000, value=8000)

if st.button("Predict Price"):
    
    input_data = pd.DataFrame({
        "location": [location],
        "bhk": [bhk],
        "bathroom": [bathroom],
        "balcony": [balcony],
        "area_sqft": [area_sqft],
        "price_per_sqft": [price_per_sqft]
    })
    
    prediction = model.predict(input_data)
    
    st.success(f"💰 Estimated House Price: ₹ {int(prediction[0]):,}")
