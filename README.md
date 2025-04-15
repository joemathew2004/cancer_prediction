# Breast Cancer Prediction Project

This project uses machine learning to predict whether a tumor is benign or malignant based on its characteristics.

## Overview

The project includes:

* **`Cancer_Data.csv`**: The dataset used for training the model.
* **`train_model.py`**: Python script to train the machine learning model (Logistic Regression) and save the trained model and scaler.
* **`predict_terminal.py`**: Python script to get feature inputs from the terminal and make a prediction using the saved model.
* **`app.py`**: A Streamlit web application for interactive predictions.
* **`breast_cancer_model.joblib`**: The saved trained machine learning model.
* **`breast_cancer_scaler.joblib`**: The saved scaler used for feature preprocessing.
* **`requirements.txt`**: Lists the Python libraries required to run the project.
* **`.gitignore`**: Specifies files that Git should ignore.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd breast-cancer-prediction
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

Run the `train_model.py` script to train the model and save the necessary files:

```bash
python train_model.py
