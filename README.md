# Breast Cancer Prediction Project

This project uses machine learning to predict whether a tumor is benign or malignant based on its characteristics.

## Overview

The project includes:

* **`Cancer_Data.csv`**: The dataset used for training the model.
* **`train.py`**: Python script to train the machine learning model (Logistic Regression) and save the trained model and scaler.
* **`cancer_pred.py`**: Python script to get feature inputs from the terminal and make a prediction using the saved model.
* **`app.py`**: A Streamlit web application for interactive predictions.
* **`breast_cancer_model.joblib`**: The saved trained machine learning model.
* **`breast_cancer_scaler.joblib`**: The saved scaler used for feature preprocessing.
* **`requirements.txt`**: Lists the Python libraries required to run the project.
* **`.gitignore`**: Specifies files that Git should ignore.

## Usage

### Training the Model

Run the `train.py` script to train the model and save the necessary files:

```bash
python train.py
```

### Terminal Prediction
Run the predict_terminal.py script to get a prediction by entering feature values in the terminal:

Bash

python predict_terminal.py
Web Application
Run the app.py script using Streamlit to interact with the web application:

Bash

streamlit run app.py
Data Source
The project uses the Breast Cancer dataset (contained in Cancer_Data.csv).

Model
A Logistic Regression model was used for prediction.

Libraries
Streamlit
Pandas
Scikit-learn
Joblib
Author
[Your Name/GitHub Username]
