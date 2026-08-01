# Requirements - pandas, scikit-learn, joblib.

# Imports the necessary libraries.
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import dump

# Adding models to the list.
models = []
models.append(('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)))
models.append(('Gaussian Naive Bayes', GaussianNB()))
models.append(('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=8)))
models.append(('Random Forest Classifier', RandomForestClassifier(random_state=42)))
models.append(('Support Vector Machine', SVC(kernel='linear')))


# Load datasets.
x_train = pd.read_csv("data/x_train.csv")
y_train = pd.read_csv("data/y_train.csv")
x_test = pd.read_csv("data/x_test.csv")
y_test = pd.read_csv("data/y_test.csv")



# Training each model and saving the trained model.
for name, model in models:
    
    # Training the model.
    print(f"Training {name}...")# Displays the name of the model being trained.
    model.fit(x_train, y_train)
    
    # Saving the trained model.
    dump(model, f"results/{name.replace(' ', '_').lower()}.joblib")
