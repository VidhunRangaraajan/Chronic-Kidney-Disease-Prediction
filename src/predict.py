# Requirements - pandas, joblib.

# Importing necessary libraries.
import pandas as pd
from joblib import load

# Paths to the trained models.
MODEL_FILES = {
    "Extra Trees Classifier": "results/extra_trees_classifier.joblib",
    "LightGBM": "results/lightgbm.joblib",
    "MLP Classifier": "results/mlp_classifier.joblib",
    "Support Vector Machine": "results/support_vector_machine.joblib",
    "XGBoost": "results/xgboost.joblib",
}

# Path of scaler.
SCALER_FILE = "data/scaler.joblib"

# Columns that were standardized.
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

# Module-level caches so models/scaler are loaded from disk only once,
# even if predict_mode() is called many times from another file.
_models = None
_scaler = None


def _load_models():
    """Loads and caches the 5 trained models."""
    global _models
    if _models is None:
        _models = {name: load(path) for name, path in MODEL_FILES.items()}
    return _models


def _load_scaler():
    """Loads and caches the fitted StandardScaler."""
    global _scaler
    if _scaler is None:
        _scaler = load(SCALER_FILE)
    return _scaler


def preprocess(raw_data):
    
    df = pd.DataFrame([raw_data]) if isinstance(raw_data, dict) else raw_data.copy()
    
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

    return df


def predict_mode(input_data, already_preprocessed=False):
    
    df = input_data if already_preprocessed else preprocess(input_data)

    models = _load_models()
    predictions = pd.DataFrame({name: model.predict(df) for name, model in models.items()})
    predictions_list = predictions.values.tolist()
    y_mode = predictions.mode(axis=1)[0].astype(int).tolist()

    if len(y_mode) == 1:
        if y_mode[0] == 1:
            return ["CKD",] + predictions_list[0]
        return ["Not CKD",] + predictions_list[0]
    else:
        if y_mode[0] == 1:
            return ["CKD",] + predictions_list[0]
        return ["Not CKD",] + predictions_list[0]

if __name__ == "__main__":
    # Quick manual test using the n'th row row of the saved test set.
    n = int(input("Enter the row number of the test set to predict (0-100): "))
    x_test = pd.read_csv("data/x_test.csv")
    result = predict_mode(x_test.iloc[[n-2]], already_preprocessed=True)
    print(f"Prediction: {result}")