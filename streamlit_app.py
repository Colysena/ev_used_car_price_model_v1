# streamlit_app.py

import os
import streamlit as st
import pandas as pd
import joblib

# 0) Page config + CSS to enlarge metric labels & values
st.set_page_config(
    page_title="EV Used Car Price Prediction Comparison",
    layout="wide",
)
st.markdown(
    """
    <style>
      [data-testid="stMetricLabel"] {
        font-size: 20px !important;    /* larger model name */
      }
      [data-testid="stMetricValue"] {
        font-size: 32px !important;    /* larger price */
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ EV Used Car Price Prediction Comparison")

# 1) Load your bundled preprocessor + models
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all_models.pkl")
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Could not find `all_models.pkl` at:\n  {MODEL_PATH}")
    st.stop()

try:
    all_objects = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load `all_models.pkl`:\n{e}")
    st.stop()

if "preprocessor" not in all_objects:
    st.error("❌ Your pickle is missing the key `'preprocessor'`.")
    st.stop()
preprocessor = all_objects["preprocessor"]

model_names = [
    "LinearRegression", "Ridge",
    "DecisionTree", "RandomForest", "XGBoost", "KNN"
]
missing = [m for m in model_names if m not in all_objects]
if missing:
    st.error(f"❌ These models are missing in your pickle: {missing}")
    st.stop()

# 2) Layout: inputs on left, results on right
col1, col2 = st.columns(2)

with col1:
    st.header("Enter Car Features")
    with st.form("input_form"):
        color = st.selectbox(
            "1. Color",
            ["Traditional", "Non-traditional"],
            help=(
                'If the car color is White, Black, Grey, or Silver, please select "Traditional." '
                'If the car color is any other color, please select "Non-traditional."'
            )
        )
        year = st.slider("2. Manufactured Year", 2016, 2025, 2020, 1)
        mileage = st.number_input("3. Mileage (km)", min_value=0, value=50000, step=1000)
        latest_msrp = st.number_input("4. Latest MSRP", min_value=0, value=30000, step=1000)
        car_type = st.selectbox(
            "5. Car Type",
            ["sedan", "hatchback", "coupe", "SUV", "van", "station wagon"]
        )
        submitted = st.form_submit_button("🔍 Compare Prices")

with col2:
    st.header("Predicted Prices")
    if not submitted:
        st.write("Fill out the form on the left and click **Compare Prices** to see results.")
    else:
        # Prepare input DataFrame
        input_df = pd.DataFrame([{
            "Adjusted_color":    color,
            "Manufactured_year": year,
            "Mileage (km)":      mileage,
            "Latest MSRP":       latest_msrp,
            "Type of Car":       car_type
        }])

        # Preprocess
        try:
            X_trans = preprocessor.transform(input_df)
        except Exception as e:
            st.error(f"❌ Preprocessing failed:\n{e}")
            st.stop()

        # Predict
        raw_preds = {}
        for name in model_names:
            mdl = all_objects[name]
            try:
                raw_preds[name] = mdl.predict(X_trans)[0]
            except Exception:
                raw_preds[name] = None

        # Descriptions for help tooltips
        descriptions = {
            "LinearRegression": (
                "It finds the best straight line that shows the relationship between features "
                "(like car age, mileage, brand) and the EV’s price.\n\n"
                "In EV used car price prediction: It predicts the price based on simple patterns, "
                "like \"older cars have lower prices\" in a straight-line relationship."
            ),
            "Ridge": (
                "It works like Linear Regression but adds a penalty if the model tries to fit the data too perfectly.\n\n"
                "In EV used car price prediction: It helps when many features (like mileage and battery health) "
                "are related, making the model more stable and preventing overfitting."
            ),
            "DecisionTree": (
                "It splits the data by asking questions like \"Is mileage > 50,000 km?\" or \"Is battery health > 80%?\" "
                "and makes a decision at each branch.\n\n"
                "In EV used car price prediction: It can create simple rules, like "
                "\"If battery health is low, price drops significantly.\""
            ),
            "RandomForest": (
                "It builds many different Decision Trees on random parts of the data and averages their results "
                "to make better predictions.\n\n"
                "In EV used car price prediction: It can capture more complex factors, like combining brand, mileage, "
                "and battery warranty together to predict the price more accurately."
            ),
            "XGBoost": (
                "It builds trees step-by-step, where each new tree focuses on the mistakes of the previous ones, "
                "making the final model very accurate.\n\n"
                "In EV used car price prediction: It can find hidden patterns, like "
                "\"Tesla cars hold value better after 3 years if they have free supercharging,\" "
                "and adjust the price prediction accordingly."
            ),
            "KNN": (
                "It finds the most similar cars (neighbors) based on features like mileage, year, and battery condition, "
                "and predicts the price based on those similar cars.\n\n"
                "In EV used car price prediction: It predicts the price by looking at prices of other EVs that are most similar "
                "to the one being evaluated."
            ),
        }

        # Split into two groups
        linear_names    = ["LinearRegression", "Ridge"]
        nonlinear_names = ["RandomForest", "DecisionTree", "XGBoost", "KNN"]

        lin_col, nonlin_col = st.columns(2)

        with lin_col:
            st.subheader("Linear Regression Models")
            for name in linear_names:
                val = raw_preds.get(name)
                disp = f"{val:,.0f}" if val is not None else "Error"
                st.metric(name, disp, help=descriptions[name])

        with nonlin_col:
            st.subheader("Non-Linear Regression Models")
            for name in nonlinear_names:
                val = raw_preds.get(name)
                disp = f"{val:,.0f}" if val is not None else "Error"
                st.metric(name, disp, help=descriptions[name])

        # 4) MSE Table
        st.subheader("Model Root Mean Squared Error (RMSE)")
        mse_dict = {
            "LinearRegression": 123_499,
            "Ridge":            125_199,
            "RandomForest":      64_611,
            "XGBoost":          118_827,
            "DecisionTree":      81_309,
            "KNN":              397_126,
        }
        mse_df = pd.DataFrame.from_dict(
            mse_dict, orient="index", columns=["RMSE"]
        )
        mse_df.index.name = "Model"
        mse_df["RMSE"] = mse_df["RMSE"].map(lambda x: f"{x:,}")
        st.table(mse_df)
