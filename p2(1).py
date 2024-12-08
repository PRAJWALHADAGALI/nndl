import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Breast Cancer dataset
breast_cancer_sklearn = load_breast_cancer()
breast_cancer_df = pd.DataFrame(data=breast_cancer_sklearn.data,
                                columns=breast_cancer_sklearn.feature_names)
breast_cancer_df['target'] = breast_cancer_sklearn.target

# Separate features and target
x = breast_cancer_df.iloc[:, :-1]
y = breast_cancer_df.iloc[:, -1]

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize and train the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(5, 3), activation='relu', solver='lbfgs', random_state=42)
mlp.fit(x_train, y_train)

# Predict the test set
y_pred = mlp.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

class_report = classification_report(y_test, y_pred, zero_division=0)
print("Classification Report:\n", class_report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(pd.DataFrame(conf_matrix, 
                   index=["Actual 0", "Actual 1"], 
                   columns=["Predicted 0", "Predicted 1"]))
