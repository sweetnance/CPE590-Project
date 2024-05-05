# -*- coding: utf-8 -*-
"""
Created on Sat May  4 08:50:54 2024

@author: aas0041
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset assuming the first row is the header
logistics_modell_path = 'C:/Users/aas0041/Desktop/590 Project/socioeconomic_data.xlsx'
logistics_modell = pd.read_excel(logistics_modell_path, header=0)

# Specify selected columns for logistic regression
selected_columns = ['EP_POV150', 'EP_UNEMP', 'EP_HBURD', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
                    'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_AFAM', 'EP_HISP', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 
                    'EP_NOVEH', 'EP_GROUPQ']

# Selecting features and target variable
X = logistics_modell[selected_columns]
y = logistics_modell['Vulnerability']

# Split the dataset into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Instantiate the Logistic Regression model
logreg = LogisticRegression(random_state=16, max_iter=1000)

# Fit the model with training data
logreg.fit(X_train, y_train)

# Predict on the test dataset
y_pred = logreg.predict(X_test)

# Calculate accuracy as a percentage using the test dataset
accuracy_percentage = logreg.score(X_test, y_test) * 100
print(f"Accuracy: {accuracy_percentage:.2f}%")
# Write results to console with explanations
print(f"Accuracy: The model achieved an accuracy of {accuracy_percentage:.2f}% on the test dataset.\n")

# Generate the confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cnf_matrix)

# Visualize the Confusion Matrix as a Heatmap
class_names = np.unique(y)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix Heatmap (Normal model)')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Count of samples correctly classified for each class
correct_classifications = np.diag(cnf_matrix)
for class_name, count in zip(class_names, correct_classifications):
    print(f"Number of samples correctly classified for {class_name}: {count}")

print("Confusion Matrix:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {correct_classifications[i]} out of {np.sum(cnf_matrix[i])} samples were correctly classified.")
print()

# Generate and print the classification report
class_report = classification_report(y_test, y_pred, target_names=class_names)
print("\nClassification Report:")
print(class_report)
