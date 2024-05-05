# -*- coding: utf-8 -*-
"""
Created on Sat May  4 09:07:29 2024

@author: aas0041
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
logistics_model_path = 'C:/Users/aas0041/Desktop/590 Project/socioeconomic_data.xlsx'
logistics_model = pd.read_excel(logistics_model_path, header=0)

# Specify selected columns for logistic regression
selected_columns = ['EP_POV150', 'EP_UNEMP', 'EP_HBURD', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
                    'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_AFAM', 'EP_HISP', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 
                    'EP_NOVEH', 'EP_GROUPQ']

# Selecting features and target variable
X = logistics_model[selected_columns].values
y = logistics_model['Vulnerability'].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Logistic Regression with hyperparameter tuning
logistic_model = LogisticRegression(random_state=16, max_iter=1000)
parameters = {'C': np.logspace(-4, 4, 20), 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
grid_search = GridSearchCV(estimator=logistic_model, param_grid=parameters, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
best_logreg = grid_search.best_estimator_
accuracy_percentage = best_logreg.score(X_test, y_test) * 100
print(f"Test Accuracy after tuning: {accuracy_percentage:.2f}%")

# Random Forest as an alternative model
rf_model = RandomForestClassifier(n_estimators=100, random_state=16)
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test) * 100
print(f"Random Forest Test Accuracy: {rf_accuracy:.2f}%")

# Choose model based on performance
if rf_accuracy > accuracy_percentage:
    final_model = rf_model
    y_pred = rf_model.predict(X_test)
    print("Using Random Forest.")
else:
    final_model = best_logreg
    y_pred = best_logreg.predict(X_test)
    print("Using Tuned Logistic Regression.")

# Generate and print the confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cnf_matrix)

# Visualize the Confusion Matrix as a Heatmap
class_names = logistics_model['Vulnerability'].unique()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix Heatmap (Hyperparameter tuning)')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Count of samples correctly classified for each class
correct_classifications = np.diag(cnf_matrix)
for class_name, count in zip(class_names, correct_classifications):
    print(f"Number of samples correctly classified for {class_name}: {count}")

# Generate and print the classification report
class_report = classification_report(y_test, y_pred, target_names=class_names)
print("\nClassification Report:")
print(class_report)
