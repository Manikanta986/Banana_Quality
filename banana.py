import streamlit as st
import pandas as pd
import pickle

# ---- LOAD MODEL ----
with open("Banana.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🍌 Banana Quality Prediction")

# ---- NUMERICAL INPUTS ----
quality_score = st.number_input("quality_score", min_value=0.0, value=5.0, step=0.01)
ripeness_index = st.number_input("ripeness_index", min_value=0.0, value=20.0, step=0.1)
sugar_content_brix = st.number_input("sugar_content_brix", min_value=0.0, value=45.0, step=0.1)
firmness_kgf = st.number_input("firmness_kgf", min_value=0.0, value=10.0, step=0.01)
length_cm = st.number_input("length_cm", min_value=0.0, value=35.0, step=0.01)
weight_g = st.number_input("weight_g", min_value=0.0, value=350.0, step=0.01)
tree_age_years = st.number_input("tree_age_years", min_value=0.0, value=30.0, step=0.01)
altitude_m = st.number_input("altitude_m", min_value=0.0, value=1700.0, step=0.01)
rainfall_mm = st.number_input("rainfall_mm", min_value=0.0, value=3500.0, step=0.01)
soil_nitrogen_ppm = st.number_input("soil_nitrogen_ppm", min_value=0.0, value=200.0, step=0.01)

# ---- CATEGORICAL INPUTS ----
variety = st.selectbox(
    "variety",
    ['Manzano', 'Plantain', 'Burro', 'Red Dacca', 'Fehi',
     'Lady Finger', 'Blue Java', 'Cavendish']
)

region = st.selectbox(
    "region",
    ['Colombia', 'Guatemala', 'Ecuador', 'Costa Rica',
     'Brazil', 'Honduras', 'India', 'Philippines']
)

ripeness_category = st.selectbox(
    "ripeness_category",
    ['Green', 'Turning', 'Ripe', 'Overripe']
)

# ---- ENCODING MAPS (MUST MATCH TRAINING) ----
variety_map = {
    'Manzano': 0,
    'Plantain': 1,
    'Burro': 2,
    'Red Dacca': 3,
    'Fehi': 4,
    'Lady Finger': 5,
    'Blue Java': 6,
    'Cavendish': 7
}

region_map = {
    'Colombia': 0,
    'Guatemala': 1,
    'Ecuador': 2,
    'Costa Rica': 3,
    'Brazil': 4,
    'Honduras': 5,
    'India': 6,
    'Philippines': 7
}

ripeness_category_map = {
    'Green': 0,
    'Turning': 1,
    'Ripe': 2,
    'Overripe': 3
}

# ---- CREATE INPUT DATAFRAME ----
input_data = pd.DataFrame([{
    "quality_score": quality_score,
    "ripeness_index": ripeness_index,
    "sugar_content_brix": sugar_content_brix,
    "firmness_kgf": firmness_kgf,
    "length_cm": length_cm,
    "weight_g": weight_g,
    "tree_age_years": tree_age_years,
    "altitude_m": altitude_m,
    "rainfall_mm": rainfall_mm,
    "soil_nitrogen_ppm": soil_nitrogen_ppm,
    "variety": variety_map[variety],
    "region": region_map[region],
    "ripeness_category": ripeness_category_map[ripeness_category]
}])

# ---- IMPORTANT: FEATURE ORDER MUST MATCH TRAINING ----
input_data = input_data[[
    "quality_score",
    "ripeness_index",
    "sugar_content_brix",
    "firmness_kgf",
    "length_cm",
    "weight_g",
    "tree_age_years",
    "altitude_m",
    "rainfall_mm",
    "soil_nitrogen_ppm",
    "variety",
    "region",
    "ripeness_category"
]]

# ---- PREDICTION ----
if st.button("Predict Banana Quality"):

    prediction = model.predict(input_data)[0]

    st.subheader("🍌 Prediction Result")

    if prediction == "Premium":
        st.success("🌟 Premium Quality Banana")

    elif prediction == "Good":
        st.info("👍 Good Quality Banana")

    elif prediction == "Processing":
        st.warning("⚙️ Processing Quality Banana")

    elif prediction == "Unripe":
        st.error("🍃 Unripe Banana")

    else:
        st.error(f"Unknown class returned: {prediction}")
