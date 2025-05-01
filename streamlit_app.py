# streamlit_app.py

import os
import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb  # for loading XGBoost JSON

# 0) Page config + CSS
st.set_page_config(
    page_title="EV Used Car Price Prediction Comparison",
    layout="wide",
)
st.markdown(
    """
    <style>
      [data-testid="stMetricLabel"] {
        font-size: 20px !important;
      }
      [data-testid="stMetricValue"] {
        font-size: 32px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("⚡ EV Used Car Price Prediction Comparison")

# 1) Load sklearn models + preprocessor from pickle
PICKLE_PATH = os.path.join("models", "all_models.pkl")
if not os.path.exists(PICKLE_PATH):
    st.error(f"❌ Could not find `all_models.pkl` at {PICKLE_PATH}")
    st.stop()

all_objects = joblib.load(PICKLE_PATH)
if "preprocessor" not in all_objects:
    st.error("❌ `preprocessor` key not found in pickle.")
    st.stop()
preprocessor = all_objects["preprocessor"]

# 2) Load XGBoost from JSON
XGB_JSON = os.path.join("models", "xgb_model.json")
if not os.path.exists(XGB_JSON):
    st.error(f"❌ Could not find `xgb_model.json` at {XGB_JSON}")
    st.stop()
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGB_JSON)
all_objects["XGBoost"] = xgb_model

# define model groups
linear_names    = ["LinearRegression", "Ridge"]
nonlinear_names = ["RandomForest", "DecisionTree", "XGBoost", "KNN"]

# define brands (must match training)
BRANDS = [
    "MG", "Neta", "BYD", "Ora", "Aion", "Wuling",
    "VOLT", "Tesla", "Ford", "NISSAN", "Hyundai",
    "Audi", "Jaguar", "Mini", "Benz", "Porsche",
    "Volvo", "BMW"
]

# 3) Build UI: inputs on left, results on right
col1, col2 = st.columns(2)

with col1:
    st.header("Enter Car Features")
    with st.form("input_form"):
        color = st.selectbox(
            "1. Color",
            ["Traditional", "Non-traditional"],
            help=(
                'If the car color is White, Black, Grey, or Silver, select "Traditional." '
                'Otherwise, select "Non-traditional."'
            )
        )
        year = st.slider("2. Manufactured Year", 2016, 2025, 2020, 1)
        mileage = st.number_input("3. Mileage (km)", min_value=0, value=50000, step=1000)
        latest_msrp = st.number_input("4. Latest MSRP", min_value=0, value=30000, step=1000)
        car_type = st.selectbox(
            "5. Car Type",
            ["sedan", "hatchback", "coupe", "SUV", "van", "station wagon"]
        )
        brand = st.selectbox(
            "6. Brand",
            BRANDS
        )
        submitted = st.form_submit_button("🔍 Compare Prices")

with col2:
    st.header("Predicted Prices")
    if not submitted:
        st.write("Fill out the form on the left and click **Compare Prices** to see results.")
    else:
        # Assemble input DataFrame (including Brand)
        input_df = pd.DataFrame([{
            "Adjusted_color":    color,
            "Manufactured_year": year,
            "Mileage (km)":      mileage,
            "Latest MSRP":       latest_msrp,
            "Type of Car":       car_type,
            "Brand":             brand
        }])

        # Preprocess
        try:
            X_trans = preprocessor.transform(input_df)
        except Exception as e:
            st.error(f"❌ Preprocessing failed:\n{e}")
            st.stop()

        # Predict
        raw_preds = {}
        for name in linear_names + nonlinear_names:
            mdl = all_objects.get(name)
            try:
                raw_preds[name] = mdl.predict(X_trans)[0]
            except Exception:
                raw_preds[name] = None

        # Descriptions for tooltips (abbreviated here)
        descriptions = {
            "LinearRegression": "Straight-line fit between features and price.",
            "Ridge":            "Linear with penalty to prevent overfit.",
            "RandomForest":     "Ensemble of trees averaged.",
            "DecisionTree":     "Rule-based splits on features.",
            "XGBoost":          "Sequential trees correcting errors.",
            "KNN":              "Averages k most similar cars."
        }

        # Display side-by-side
        lin_col, nonlin_col = st.columns(2)

        with lin_col:
            st.subheader("Linear Models")
            for name in linear_names:
                val = raw_preds[name]
                disp = f"{val:,.0f}" if val is not None else "Error"
                st.metric(name, disp, help=descriptions.get(name, ""))

        with nonlin_col:
            st.subheader("Non-Linear Models")
            for name in nonlinear_names:
                val = raw_preds[name]
                disp = f"{val:,.0f}" if val is not None else "Error"
                st.metric(name, disp, help=descriptions.get(name, ""))

        # 4) RMSE + SD RMSE Table (sorted by RMSE)
        st.subheader("Model RMSE and SD RMSE")
        rmse_data = {
            "Linear Regression":   {"RMSE": 291561.32, "SD RMSE": 35911.37},
            "Ridge Regression":    {"RMSE": 291089.36, "SD RMSE": 32327.44},
            "Random Forest":       {"RMSE": 265440.26, "SD RMSE": 52327.85},
            "Gradient Boosting":   {"RMSE": 264073.08, "SD RMSE": 49514.23},
            "Decision Tree":       {"RMSE": 342104.69, "SD RMSE": 26443.26},
            "KNN Regressor":       {"RMSE": 283916.80, "SD RMSE": 38716.82},
        }
        rmse_df = pd.DataFrame.from_dict(rmse_data, orient="index")
        rmse_df.index.name = "Model"
        rmse_df = rmse_df.sort_values(by="RMSE", ascending=True)
        rmse_df["RMSE"]    = rmse_df["RMSE"].map(lambda x: f"{x:,.2f}")
        rmse_df["SD RMSE"] = rmse_df["SD RMSE"].map(lambda x: f"{x:,.2f}")
        st.table(rmse_df)
