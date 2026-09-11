# Banking77 Intent Classifier

A machine learning project for classifying banking customer queries into 77 intent categories using the Banking77 dataset.

## Project Overview

The goal of this project is to automatically identify the intent behind a customer's banking-related message.

The project includes:

- exploratory data analysis (EDA)
- text preprocessing
- TF-IDF feature extraction
- Logistic Regression baseline
- comparison with Linear SVM
- hyperparameter tuning using GridSearchCV
- TF-IDF experiments
- error analysis
- an interactive Streamlit application for real-time predictions

---

## Dataset

This project uses the Banking77 dataset, designed for banking intent classification.

The dataset contains customer queries grouped into 77 different intent categories.

Examples of intents include:

- `card_payment_not_recognised`
- `cash_withdrawal_not_recognised`
- `declined_card`
- `cash_withdrawal`
- `card_payment_fee_charged`
- `cash_withdrawal_charge`
- `refund_not_showing_up`

The dataset is split into:

- Training set: 10,003 samples
- Test set: 3,080 samples
- Number of intent classes: 77

Each sample contains a customer message and its corresponding intent label.

---

## Exploratory Data Analysis

The exploratory analysis included:

- checking dataset shape and columns
- checking missing values
- checking duplicate samples
- analysing the number of intent classes
- analysing class distribution
- analysing text length
- inspecting examples from different intent categories
- analysing model classification errors

The dataset contains many semantically similar banking intents, which makes the classification task challenging.

For example, the model needs to distinguish between intents related to:

- card payments
- cash withdrawals
- transfers
- refunds
- card issues

Some customer messages may contain similar vocabulary while referring to different banking operations.

---

## Modeling

A classical NLP pipeline was used for intent classification:

```text
Text
  ↓
TF-IDF Vectorization
  ↓
Logistic Regression
  ↓
Predicted Intent
```

Several experiments were performed during model development:

- Logistic Regression as the baseline model
- hyperparameter tuning of Logistic Regression using `GridSearchCV`
- Linear SVM (`LinearSVC`) for comparison
- TF-IDF experiments with different parameters
- bigram features
- different `min_df` values
- sublinear TF-IDF scaling

The final model uses:

```python
TfidfVectorizer(
    stop_words="english",
    min_df=2
)
```

and:

```python
LogisticRegression(
    C=10,
    max_iter=1000
)
```

The vectorizer and classifier are combined into a Scikit-learn `Pipeline`.

This allows the application to work directly with raw text without manually transforming text before making predictions.

---

## Model Comparison

| Model / Configuration | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression baseline | 0.8552 | 0.8548 |
| Linear SVM | 0.8500 | 0.8485 |
| TF-IDF + bigrams | 0.8494 | 0.8490 |
| Final TF-IDF (`min_df=2`) + Logistic Regression | **0.8565** | **0.8562** |

The final Logistic Regression pipeline achieved the best overall performance among the tested classical machine learning approaches.

---

## Final Model Performance

The final model achieved:

```text
Accuracy: 0.8565
Macro F1: 0.8562
```

Macro F1 was used together with accuracy because the project contains 77 intent classes and performance across all classes is important.

---

## Error Analysis

After training the models, prediction errors were analysed to understand which intent categories were most difficult to distinguish.

Many errors occurred between semantically similar banking intents.

For example, messages about:

- unrecognised card payments
- unrecognised cash withdrawals
- refunds
- declined transactions
- different types of transfers

can contain similar words and sentence structures.

This analysis showed that some classification errors are caused not only by the model but also by the semantic similarity between intent categories.

---

## Prediction Pipeline

The trained model accepts raw customer text:

```text
"Why was my card declined?"
```

and returns the predicted intent:

```text
declined_card
```

The application can also return the Top-3 most probable intents.

Example:

```Top-3 predictions:
declined_card
card_payment_wrong_exchange_rate
cash_withdrawal
```

This makes it possible to inspect alternative model predictions when the classification is uncertain.

---

## Streamlit Application

An interactive Streamlit application was created for testing the trained model.

The application allows the user to:

- enter a banking-related customer message
- receive the predicted intent
- view the Top-3 predictions
- view the probability assigned to each prediction

Live demo:

[Banking77 Intent Classifier](https://portfolio-1-project-banking77-intent-classifier.streamlit.app/)


## Project Structure

```text
banking77-intent-classifier/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── banking77_pipeline.joblib
│
├── notebooks/
│   └── NLP_pet_project_Askhat_Meruyert.ipynb
│
└── src/
    ├── train.py
    └── predict.py
```

### Main files

`notebooks/`

Contains exploratory data analysis, experiments, model comparison and error analysis.

`src/train.py`

Loads the dataset, creates the machine learning pipeline, trains the model, evaluates it and saves the trained pipeline.

`src/predict.py`

Contains functions for loading the trained model and making predictions.

`app.py`

Contains the Streamlit user interface.

`models/`

Contains the trained Scikit-learn pipeline.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AskhatMeruyert/banking77-intent-classifier.git
cd banking77-intent-classifier
```

Create or activate a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

---

## Training the Model

To train the model again, run:

```bash
python src/train.py
```

The script will:

```text
Load training and test data
        ↓
Create TF-IDF + Logistic Regression Pipeline
        ↓
Train the model
        ↓
Evaluate Accuracy and Macro F1
        ↓
Save the trained model
```

The trained pipeline is saved to:

```text
models/banking77_pipeline.joblib
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then enter a banking-related message into the text field and click **Predict**.

---

## Technologies

The project was built using:

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

---

## Limitations

The model is based on TF-IDF features and therefore primarily relies on words and their statistical importance.

It does not understand language context as deeply as transformer-based models such as BERT.

Some Banking77 categories are also semantically very similar, which can cause confusion between related intents.

---

## Future Improvements

Possible future improvements include:

- testing transformer-based models such as BERT or DistilBERT
- adding confidence thresholds for uncertain predictions
- improving the Streamlit interface
- deploying the application online
- containerising the application with Docker
- analysing predictions on custom real-world banking messages

---

## Author

**Meruyert Askhat**

Machine Learning / NLP pet project.

## Live Demo

Try the deployed application here:

[Open Streamlit App](https://portfolio-1-project-banking77-intent-classifier.streamlit.app/)