import joblib

MODEL_PATH = "models/banking77_pipeline.joblib"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def predict_intent(text, model):
    prediction = model.predict([text])[0]
    return prediction

def predict_top3(text, model):
    probabilities = model.predict_proba([text])[0]

    classes = model.named_steps["model"].classes_

    top3_indices = probabilities.argsort()[-3:][::-1]

    results = []

    for index in top3_indices:
        results.append({
            "intent": classes[index],
            "probability": float(probabilities[index])
        })

    return results

if __name__ == "__main__":

    
    model = load_model()

    text = "Why was my card declined?"

    prediction = predict_intent(text, model)

    print("Text:", text)
    print("Predicted intent:", prediction)

    top3 = predict_top3(text, model)

    print("\nTop-3 predictions:")

    for result in top3:
        print(
            result["intent"],
            f"{result['probability'] * 100:.2f}%"
    )