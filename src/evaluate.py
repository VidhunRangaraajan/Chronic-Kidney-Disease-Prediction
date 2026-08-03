# evaluate.py
# Description: Takes evaluation metrics, correlation heatmap and confusion matrix of the trained models and mode of the predictions and saves it in a CSV file and PNG files.
# Author: Vidhun Rangaraajan J
# Website: https://www.vidhun.com
# Github: https://github.com/VidhunRangaraajan
# Repository: https://github.com/VidhunRangaraajan/Chronic-Kidney-Disease-Prediction
# Requirements: pandas, seaborn, matplotlib, joblib, scikit-learn.
# Usage: To evaluate the performance of the trained models.
# Depends On: -
# Input Files: data/x_test.csv, data/y_test.csv, results/y_mode.csv, data/cleaned_kidney_disease.csv, results/extra_trees_classifier.joblib, results/lightgbm.joblib, results/mlp_classifier.joblib, results/support_vector_machine.joblib, results/xgboost.joblib
# Output Files: results/metrics.csv, results/correlation_heatmap.png, results/mode_pred_confusion_matrix.png, results/{model name}_confusion_matrix.png
# Notes: Run the preprocessing.py, followed by train_model.py and find_mode_of_predicted.py first.
# To Do: -

# Importing necessary libraries.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Loading the datasets.
x_test = pd.read_csv("data/x_test.csv")
y_test = pd.read_csv("data/y_test.csv")
y_mode = pd.read_csv("results/y_mode.csv")
df = pd.read_csv("data/cleaned_kidney_disease.csv")

# Plotting the correlation heatmap of the cleaned dataset.
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("results/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

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

# Evaluating the models.
metrics = []
for name, model in models.items():
    
    # Printing the name of the model being evaluated and finding the predicted value for the test dataset.
    print(f"Evaluating {name}...")
    y_pred = model.predict(x_test)

    # Taking evaluation metrics.
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)
    metrics.append([name, acc, prec, rec, f1, "\n" + report])

    # Finding Confusion Matrix Heatmap.
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"results/{name}_confusion_matrix.png")
    plt.close()

# Printing the name of the model being evaluated.
print("Evaluating mode of predictions...")

# Taking evaluation metrics.
acc = accuracy_score(y_test, y_mode)
prec = precision_score(y_test, y_mode, average="weighted")
rec = recall_score(y_test, y_mode, average="weighted")
f1 = f1_score(y_test, y_mode, average="weighted")
report = classification_report(y_test, y_mode)
metrics.append(['mode of pred', acc, prec, rec, f1, "\n" + report])

# Finding Confusion Matrix Heatmap.
cm = confusion_matrix(y_test, y_mode)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Mode Of Predictions")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/mode_pred_confusion_matrix.png")
plt.close()

# Saving the metrics to a CSV file.
df = pd.DataFrame(metrics, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Classification Report"])
df.to_csv("results/metrics.csv", index=False)