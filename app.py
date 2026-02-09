import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("sleep_model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Sleep Quality Predictor", layout="centered")

st.title("😴 Sleep Quality Predictor")
st.write("Predict your sleep quality based on daily lifestyle habits")

# Inputs
sleep_duration = st.number_input("Sleep Duration (hours)", 0.0, 12.0, 7.0)
screen_time = st.number_input("Screen Time Before Bed (minutes)", 0, 300, 30)
exercise = st.number_input("Exercise Duration (minutes)", 0, 180, 30)
caffeine = st.slider("Caffeine Intake (0 = None, 4 = High)", 0, 4, 1)
stress = st.slider("Stress Level (0–10)", 0, 10, 5)

if st.button("Predict Sleep Quality"):
    input_data = np.array([[sleep_duration, screen_time, exercise, caffeine, stress]])
    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)[0]

    st.subheader(f"🛌 Predicted Sleep Quality: **{result}**")

    if result == "Poor":
        st.warning("⚠️ Tips to Improve Sleep:")
        st.write("- Reduce screen time before bed")
        st.write("- Avoid caffeine at night")
        st.write("- Try relaxation techniques")
    elif result == "Average":
        st.info("🙂 Tips:")
        st.write("- Maintain consistent sleep schedule")
        st.write("- Increase physical activity")
    else:
        st.success("🎉 Great! Keep maintaining healthy sleep habits")
