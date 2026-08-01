import pandas as pd
from joblib import load
import statistics


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

ETC = predict(models["Extra Trees Classifier"])
LGBM = predict(models["Lightgbm"])
MLP = predict(models["Mlp Classifier"])
SVM = predict(models["Support Vector Machine"])
XGB = predict(models["Xgboost"])

y_mode = []
for i in range(len(ETC)):
    j=statistics.mode([ETC[i],LGBM[i],MLP[i],SVM[i],XGB[i]])
    y_mode.append(int(j))
    
df = pd.DataFrame(y_mode, columns=["Prediction"])
df.to_csv("results/y_mode.csv", index=False)