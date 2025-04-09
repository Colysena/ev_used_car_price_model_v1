# streamlit_app.py

import os
import streamlit as st
import pandas as pd
import joblib

# ——————————————————————————————
# 0) Page config
# ——————————————————————————————
st.set_page_config(
    page_title="EV Used Car Price Comparison",
    layout="wide",
)

st.title("⚡ EV Used Car Price Comparison")

# ——————————————————————————————
# 1) Load your bundled preprocessor + models
# ——————————————————————————————
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all_models.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Could not find `all_models.pkl` at:\n  {MODEL_PATH}")
    st.stop()

try:
    all_objects = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load `all_models.pkl`:\n{e}")
    st.stop()

# pull out the preprocessor
if "preprocessor" not in all_objects:
    st.error("❌ Your pickle is missing the key `'preprocessor'`.")
    st.stop()
preprocessor = all_objects["preprocessor"]

# list of model keys
model_names = [
    "LinearRegression",
    "Ridge",
    "RandomForest",
    "XGBoost",
    "DecisionTree",
    "KNN"
]
missing = [m for m in model_names if m not in all_objects]
if missing:
    st.error(f"❌ These models are missing in your pickle: {missing}")
    st.stop()

# ——————————————————————————————
# 2) Split the page into two columns
# ——————————————————————————————
col1, col2 = st.columns(2)

# — Column 1: the input form
with col1:
    st.header("Enter Car Features")
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

# — Column 2: the results table
with col2:
    st.header("Predicted Prices by Model")
    if submitted:
        # build one‐row DataFrame matching your training columns
        input_df = pd.DataFrame([{
            "Adjusted_color":    color,
            "Manufactured_year": year,
            "Mileage (km)":      mileage,
            "Latest MSRP":       latest_msrp,
            "Type of Car":       car_type
        }])

        # apply preprocessing
        try:
            X_transformed = preprocessor.transform(input_df)
        except Exception as e:
            st.error(f"❌ Preprocessing failed:\n{e}")
            st.stop()

        # run predictions
        results = {}
        for name in model_names:
            mdl = all_objects[name]
            try:
                pred = mdl.predict(X_transformed)[0]
                results[name] = f"{pred:,.0f}"
            except Exception as e:
                results[name] = f"Error: {e}"

        # display table
        results_df = pd.DataFrame.from_dict(
            results, orient="index", columns=["Estimated Price"]
        )
        results_df.index.name = "Model"
        st.table(results_df)
    else:
        st.write("Fill out the form on the left and click **Compare Prices** to see results.")
