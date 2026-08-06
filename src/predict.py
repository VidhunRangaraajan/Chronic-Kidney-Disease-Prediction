# predict.py
# Description: When the function predict_mode(input_data, bool already_preprocessed) is called, it returns the final prediction and the predictions of each model in a list. The first element of the list is the final prediction("CKD"/"NOT CKD") and the next 5 elements are the predictions of each model(in 0/1) and the first model is the best model of the five.
# Author: Vidhun Rangaraajan J
# Website: https://www.vidhun.com
# Github: https://github.com/VidhunRangaraajan
# Repository: https://github.com/VidhunRangaraajan/Chronic-Kidney-Disease-Prediction
# Requirements: pandas, joblib.
# Usage: Finds the prediction of the models and final(mode of the predictions) and used in deployment.
# Depends On: -
# Input Files: results/extra_trees_classifier.joblib, results/lightgbm.joblib, results/mlp_classifier.joblib, results/support_vector_machine.joblib, results/xgboost.joblib, data/scaler.joblib
# Output Files: -
# Notes: Requires the trained models and scaler to be present in the specified paths.
# To Do: -

# Importing necessary libraries.
import pandas as pd
from joblib import load

# Dictionary of paths to the trained models with their names as keys.
MODEL_FILES = {
    "Extra Trees Classifier": "results/extra_trees_classifier.joblib",
    "LightGBM": "results/lightgbm.joblib",
    "MLP Classifier": "results/mlp_classifier.joblib",
    "Support Vector Machine": "results/support_vector_machine.joblib",
    "XGBoost": "results/xgboost.joblib",
}

# Path of scaler.
SCALER_FILE = "data/scaler.joblib"

# List of numeric columns.
NUMERIC_COLUMNS = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
    'potassium', 'haemoglobin', 'packed_cell_volume',
    'white_blood_cell_count', 'red_blood_cell_count'
]

# Full ordered list of feature columns.
FEATURE_COLUMNS = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bateria',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
    'potassium', 'haemoglobin', 'packed_cell_volume',
    'white_blood_cell_count', 'red_blood_cell_count', 'hypertension',
    'diabetes_mellitus', 'coronary_artery_disease', 'appetite',
    'pedal_edema', 'anemia'
]

# Mappings for categorical columns to numeric values.
CATEGORICAL_MAPS = {
    'red_blood_cells': {'abnormal': 0, 'normal': 1},
    'pus_cell': {'abnormal': 0, 'normal': 1},
    'pus_cell_clumps': {'present': 1, 'notpresent': 0},
    'bateria': {'present': 1, 'notpresent': 0},
    'hypertension': {'yes': 1, 'no': 0},
    'diabetes_mellitus': {'yes': 1, 'no': 0},
    'coronary_artery_disease': {'yes': 1, 'no': 0},
    'appetite': {'good': 1, 'poor': 0},
    'pedal_edema': {'yes': 1, 'no': 0},
    'anemia': {'yes': 1, 'no': 0},
}

# Module-level caches so models/scaler are loaded from disk only once, even if predict_mode() is called many times from another file.
_models = None
_scaler = None

# When the function is called for the first time, it loads and caches the 5 trained models and returns them as a dictionary.
def _load_models():
    global _models
    if _models is None:
        _models = {name: load(path) for name, path in MODEL_FILES.items()}
    return _models

# When the function is called for the first time, it loads and caches the fitted StandardScaler and returns it.
def _load_scaler():
    global _scaler
    if _scaler is None:
        _scaler = load(SCALER_FILE)
    return _scaler

# Preprocesses the input data (either a dictionary or a DataFrame) to match the format expected by the trained models and returns the preprocessed data.
def preprocess(raw_data):
    
    # Creating a DataFrame from the input data.
    df = pd.DataFrame([raw_data]) if isinstance(raw_data, dict) else raw_data.copy()
    
    # Converting numeric columns to numeric type.
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # Applying categorical to numeric mappings.
    for col, mapping in CATEGORICAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Ensuring the columns are present and in the exact order the models expect.
    df = df[FEATURE_COLUMNS]

    # Scaling numeric columns using the scaler fit during preprocessing.
    scaler = _load_scaler()
    df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])
    
    # Returning the preprocessed DataFrame ready for prediction.
    return df

# Predicts the mode of predictions from all 5 trained models for the given input data(cleaned or runcleaned) and returns the prediction result along with the predictions of each model.
def predict_mode(input_data, already_preprocessed=False):
    
    # Preprocess the input data if it hasn't been preprocessed yet.
    df = input_data if already_preprocessed else preprocess(input_data)

    # Loading the trained models and making predictions for each model.
    models = _load_models()
    predictions = pd.DataFrame({name: model.predict(df) for name, model in models.items()})
    predictions_list = predictions.values.tolist()
    y_mode = predictions.mode(axis=1)[0].astype(int).tolist()

    # Returning the prediction result based on the mode of predictions from all models and the predictions of each model.
    if len(y_mode) == 1:
        if y_mode[0] == 1:
            return ["CKD",] + predictions_list[0]
        return ["Not CKD",] + predictions_list[0]
    else:
        if y_mode[0] == 1:
            return ["CKD",] + predictions_list[0]
        return ["Not CKD",] + predictions_list[0]

# Quick manual test using the n'th row row of the saved test set.
if __name__ == "__main__":
    n = int(input("Enter the row number of the test set to predict (0-100): "))
    x_test = pd.read_csv("data/x_test.csv")
    result = predict_mode(x_test.iloc[[n-2]], already_preprocessed=True)
    print(f"Prediction: {result}")