# streamlit_app.py

import os
import streamlit as st
import pandas as pd
import joblib

# ——————————————————————————————
# 1) Load your bundled preprocessor + models
# ——————————————————————————————

# Path to your pickle (models/all_models.pkl)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all_models.pkl")

# If it's not there, stop with an error message
if not os.path.exists(MODEL_PATH):
    st.set_page_config(layout="wide")
    st.error(
        f"❌ Could not find `all_models.pkl` here:\n\n  {MODEL_PATH}\n\n"
        "Make sure you uploaded it into the `models/` folder."
    )
    st.stop()

# Try loading it
try:
    all_objects = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load `all_models.pkl`:\n\n{e}")
    st.stop()

# Extract the preprocessor
if "preprocessor" not in all_objects:
    st.error("❌ Your pickle is missing the key `'preprocessor'`.")
    st.stop()
preprocessor = all_objects["preprocessor"]

# Define the six model names you saved
model_names = [
    "LinearRegression",
    "Ridge",
    "RandomForest",
    "XGBoost",
    "DecisionTree",
    "KNN"
]

# Make sure each model is present
missing = [m for m in model_names if m not in all_objects]
if missing:
    st.error(f"❌ These models are missing in your pickle: {missing}")
    st.stop()

# ——————————————————————————————
# 2) Build the user input form
# ——————————————————————————————

st.set_page_config(page_title="EV Price Comparison", layout="wide")
st.title("⚡ EV Used Car Price Comparison")

with st.form("input_form"):
    color = st.selectbox("1. Color", ["Traditional", "Non-traditional"])
    year = st.slider("2. Manufactured Year", 2016, 2025, 2020, 1)
    mileage = st.number_input("3. Mileage (km)", min_value=0, value=50000, step=1000)
    latest_msrp = st.number_input("4. Latest MSRP", min_value=0, value=30000, step=1000)
    car_type = st.selectbox(
        "5. Car Type",
        ["sedan", "hatchback", "coupe", "SUV", "van", "station wagon"]
    )
    submitted = st.form_submit_button("🔍 Compare Prices")

# ——————————————————————————————
# 3) On submit: transform & predict
# ——————————————————————————————

if submitted:
    # 3.1 Build a one‑row DataFrame matching your train columns
    input_df = pd.DataFrame([{
        "Adjusted_color":    color,
        "Manufactured_year": year,
        "Mileage (km)":      mileage,
        "Latest MSRP":       latest_msrp,
        "Type of Car":       car_type
    }])

    # 3.2 Apply the same preprocessing
    try:
        X_transformed = preprocessor.transform(input_df)
    except Exception as e:
        st.error(f"❌ Preprocessing failed:\n\n{e}")
        st.stop()

    # 3.3 Run each model’s predict()
    results = {}
    for name in model_names:
        mdl = all_objects[name]
        try:
            pred = mdl.predict(X_transformed)[0]
            results[name] = round(float(pred), 2)
        except Exception as e:
            results[name] = f"Error: {e}"

    # 3.4 Display as a table
    results_df = pd.DataFrame.from_dict(
        results, orient="index", columns=["Estimated Price"]
    )
    results_df.index.name = "Model"

    st.subheader("Predicted Prices by Model")
    st.table(results_df)
