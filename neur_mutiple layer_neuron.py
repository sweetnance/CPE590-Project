# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:55:45 2024

@author: aas0041
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit

# Step 1: Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Define configurations to test
configurations = [(1, 50), (1, 100), (2, 50), (2, 100), (3, 50), (3, 100)]  # Tuples of (number of layers, neurons per layer)

results = []
for layers, neurons in configurations:
    # Step 4: Initialize and compile the model
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(X_train_scaled.shape[1],)))  # First hidden layer
    
    for _ in range(1, layers):
        model.add(Dense(neurons, activation='relu'))  # Additional hidden layers with variable neurons
    
    model.add(Dense(1))  # Output layer for regression
    
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    
    # Step 5: Train the model
    history = model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test), verbose=0)  # Reduced verbosity for clarity
    
    # Step 6: Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    results.append((layers, neurons, mse))

# Step 7: Print the results
for layers, neurons, mse in results:
    print(f"Configuration: {layers} Layers, {neurons} Neurons per Layer -> MSE: {mse:.4f}")

# Step 8: Prepare data for plotting
layers = [r[0] for r in results]
neurons = [r[1] for r in results]
mses = [r[2] for r in results]

# Step 9: Plot the results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(layers, neurons, mses, c=mses, cmap='viridis', s=100)
ax.set_xlabel('Number of Layers')
ax.set_ylabel('Neurons Per Layer')
ax.set_zlabel('MSE')
plt.colorbar(scatter)
plt.title('3D Plot of MSE by Configuration')
plt.show()
