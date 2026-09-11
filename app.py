import streamlit as st

from src.predict import load_model, predict_intent, predict_top3

st.title("Banking77 Intent Classifier")
st.write(
    "Classifies banking-related customer messages into one of 77 intent categories."
)

model = load_model()

text = st.text_area(
    "Enter a banking-related message:",
    placeholder="Example: Why was my card declined?"
)

if st.button("Predict"):
    if text:
        prediction = predict_intent(text, model)
        top3 = predict_top3(text, model)

        st.write("### Predicted intent")
        st.success(prediction)

        st.write("### Top-3 predictions")

        for result in top3:
            intent = result["intent"]
            probability = result["probability"]

        st.write(f"**{intent}** — {probability * 100:.2f}%")
        st.progress(probability)