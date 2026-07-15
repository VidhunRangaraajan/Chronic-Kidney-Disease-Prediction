# Requirements - pandas, seaborn, matplotlib, joblib, scikit-learn

# Importing necessary libraries.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Loading the test dataset
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
    models[model_name] = load(file)

# Evaluating the models.
metrics = []
for name, model in models.items():
    
    # Changing the file path to file name.
    name = name[8::]
    
    # Printing the name of the model being evaluated and finding the predicted value for the test dataset
    print(f"Evaluating {name}...")
    y_pred = model.predict(x_test)

    # Taking evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)
    metrics.append([name, acc, prec, rec, f1, "\n" + report])

    # Finding Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"results/{name}_confusion_matrix.png")
    plt.close()

# Saving the metrics to a CSV file
df = pd.DataFrame(metrics, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Classification Report"])
df.to_csv("results/metrics.csv", index=False)
