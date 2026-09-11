import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    return train, test


def build_pipeline():
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                min_df=2
            )
        ),
        (
            "model",
            LogisticRegression(
                C=10,
                max_iter=1000
            )
        )
    ])

    return pipeline


def evaluate_model(model, test):
    predictions = model.predict(test["text"])

    accuracy = accuracy_score(
        test["label_text"],
        predictions
    )

    macro_f1 = f1_score(
        test["label_text"],
        predictions,
        average="macro"
    )

    print("\nModel evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


def save_model(model):
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/banking77_pipeline.joblib"
    )

    print("\nModel saved successfully!")


def main():
    train, test = load_data()

    pipeline = build_pipeline()

    pipeline.fit(
        train["text"],
        train["label_text"]
    )

    print("\nModel trained successfully!")

    evaluate_model(pipeline, test)

    save_model(pipeline)


if __name__ == "__main__":
    main()