import streamlit as st
import joblib
import pandas as pd
import os


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

MODEL_PATH = "models/pipeline.joblib"


if not os.path.exists(MODEL_PATH):
    st.error("Model not found! Please train the model before running the app.")
    st.stop()

model = joblib.load(MODEL_PATH)


feature_names = model.named_steps["preprocessor"].transformers_[0][2]

st.title("🏡 House Price Prediction App")
st.write("Fill in the property details below to get an estimated market price.")


user_inputs = {}
col_left, col_right = st.columns(2)
columns = [col_left, col_right]

for i, feature in enumerate(feature_names):
    with columns[i % 2]:

        f_lower = feature.lower()

        
        if "bed" in f_lower:
            user_inputs[feature] = st.slider(feature, 0, 10, 3)

        elif "bath" in f_lower:
            user_inputs[feature] = st.slider(feature, 0, 8, 2)

        elif "sqft" in f_lower:
            user_inputs[feature] = st.number_input(feature, 200, 15000, 2000)

        elif "lot" in f_lower:
            user_inputs[feature] = st.number_input(feature, 300, 50000, 3000)

        elif "floor" in f_lower:
            user_inputs[feature] = st.slider(feature, 1.0, 4.0, 1.0)

        elif "view" in f_lower:
            user_inputs[feature] = st.slider(feature, 0, 4, 0)

        elif "condition" in f_lower:
            user_inputs[feature] = st.slider(feature, 1, 5, 3)

        elif "yr_built" in f_lower:
            user_inputs[feature] = st.number_input(feature, 1900, 2025, 1995)

        elif "yr_renovated" in f_lower:
            user_inputs[feature] = st.number_input(feature, 0, 2025, 0)

        else:
            user_inputs[feature] = st.number_input(feature, 0.0, 10000.0, 1.0)

st.write("---")


if st.button("Predict Price", use_container_width=True):
    input_df = pd.DataFrame([user_inputs], columns=feature_names)
    predicted_price = model.predict(input_df)[0]
    st.success(f"Estimated House Price: **${predicted_price:,.2f}**")

st.write("---")
st.caption("Built with Streamlit & Scikit-Learn 💙")