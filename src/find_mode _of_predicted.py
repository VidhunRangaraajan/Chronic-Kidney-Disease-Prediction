import pandas as pd
from joblib import load
import statistics


x_test = pd.read_csv("data/x_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# Saving the model's file path in a list.
files = [
    "results/decision_tree_classifier.joblib",
    "results/gaussian_naive_bayes.joblib",
    "results/k-nearest_neighbors.joblib",
    "results/random_forest_classifier.joblib",
    "results/support_vector_machine.joblib"
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

DTC = predict(models["Decision Tree Classifier"])
GNB = predict(models["Gaussian Naive Bayes"])
KNN = predict(models["K-Nearest Neighbors"])
RFC = predict(models["Random Forest Classifier"])
SVM = predict(models["Support Vector Machine"])

y_mode = []
for i in range(len(DTC)):
    j=statistics.mode([DTC[i],GNB[i],KNN[i],RFC[i],SVM[i]])
    y_mode.append(int(j))
    
df = pd.DataFrame(y_mode, columns=["Prediction"])
df.to_csv("results/y_mode.csv", index=False)