# Requirements - pandas, xgboost, lightgbm, scikit-learn, joblib.

# Imports the necessary libraries.
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from joblib import dump

# Adding models to the list.
models = []
models.append(('XGBoost', XGBClassifier(random_state=42, eval_metric='logloss')))
models.append(('LightGBM', LGBMClassifier(random_state=42, boosting_type='gbdt', num_leaves=63, reg_alpha=0.0, reg_lambda=0.0)))
models.append(('Extra Trees Classifier', ExtraTreesClassifier(random_state=42)))
models.append(('Support Vector Machine', SVC(kernel='rbf')))
models.append(('MLP Classifier', MLPClassifier(random_state=42, max_iter=500)))

# Load datasets.
x_train = pd.read_csv("data/x_train.csv")
y_train = pd.read_csv("data/y_train.csv")

# Training each model and saving the trained model.
for name, model in models:
    
    # Training the model.
    print(f"Training {name}...")# Displays the name of the model being trained.
    model.fit(x_train, y_train)
    
    # Saving the trained model.
    dump(model, f"results/{name.replace(' ', '_').lower()}.joblib")
