# find_mode_of_predicted.py
# Description: This program finds the mode of the predicted values from multiple machine learning models and saves it in a CSV file.
# Author: Vidhun Rangaraajan J
# Website: https://www.vidhun.com
# Github: https://github.com/VidhunRangaraajan
# Repository: https://github.com/VidhunRangaraajan/Chronic-Kidney-Disease-Prediction
# Requirements: pandas, joblib, statistics.
# Usage: To find the mode of the predictions.
# Depends On: -
# Input Files: data/x_test.csv, data/y_test.csv, results/extra_trees_classifier.joblib, results/lightgbm.joblib, results/mlp_classifier.joblib, results/support_vector_machine.joblib, results/xgboost.joblib
# Output Files: results/y_mode.csv
# Notes: -
# To Do: Need to add by howmuch time the models are agreeing with each other in the prediction and how much models does't agree by how much time.

# Importing the required libraries.
import pandas as pd
from joblib import load
import statistics

# Loading the test data from the CSV files.
x_test = pd.read_csv("data/x_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# Saving the model's file path in a list.
files = [
    "results/extra_trees_classifier.joblib",
    "results/lightgbm.joblib",
    "results/mlp_classifier.joblib",
    "results/support_vector_machine.joblib",
    "results/xgboost.joblib"
]

# Loading the models and saving it in a dictionary.
models = {}
for file in files:
    model_name = file.replace(".joblib", "").replace("_", " ").title()
    models[model_name[8:]] = load(file)
    
# Function to predict the target variable using the loaded model.
def predict(model):
    y_pred = model.predict(x_test)
    return y_pred

# Saving each model's prediction in a variable.
ETC = predict(models["Extra Trees Classifier"])
LGBM = predict(models["Lightgbm"])
MLP = predict(models["Mlp Classifier"])
SVM = predict(models["Support Vector Machine"])
XGB = predict(models["Xgboost"])

# Finding the mode of the predicted values from all the models.
y_mode = []
for i in range(len(ETC)):
    j=statistics.mode([ETC[i],LGBM[i],MLP[i],SVM[i],XGB[i]])
    y_mode.append(int(j))

# Saving the mode of the predicted values in a CSV file.
df = pd.DataFrame(y_mode, columns=["Prediction"])
df.to_csv("results/y_mode.csv", index=False)