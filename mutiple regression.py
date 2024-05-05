# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:21:30 2024

@author: aas0041
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the data
data = pd.read_excel('C:/Users/aas0041/Desktop/590 Project/socioeconomic_data.xlsx')

# Selecting features and target variable
X = data[['EP_POV150', 'EP_UNEMP', 'EP_HBURD', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
          'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_AFAM', 'EP_HISP', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 
          'EP_NOVEH', 'EP_GROUPQ']]
y = data['Overall SVI']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Predicting the test set results
y_pred = model.predict(X_test)

# Calculating the R² score
r2_train = model.score(X_train, y_train)
r2_test = r2_score(y_test, y_pred)

# Getting coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

print('Coefficients:\n', coefficients)
print('Coefficient of Determination (Training):', r2_train)
print('Coefficient of Determination (Testing):', r2_test)

# Assessing model's goodness
if r2_test > 0.7:
    print("This is a good model as it explains a significant amount of variance.")
else:
    print("This model may not be good enough as it does not explain a significant amount of variance.")
