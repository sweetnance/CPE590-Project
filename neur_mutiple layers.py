# -*- coding: utf-8 -*-
"""
Created on Fri May  3 20:48:20 2024

@author: aas0041
"""


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Define the range of hidden layers to test
layer_configs = [1, 2, 3, 4, 5]  # Testing 1 to 5 hidden layers

results = []
loss_histories = []  # To store training loss histories
val_loss_histories = []  # To store validation loss histories
data = []  # List to store results for DataFrame

for layers in layer_configs:
    # Step 4: Initialize and compile the model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(X_train_scaled.shape[1],)))  # First hidden layer
    
    for _ in range(1, layers):
        model.add(Dense(50, activation='relu'))  # Additional hidden layers
    
    model.add(Dense(1))  # Output layer for regression
    
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    
    # Step 5: Train the model
    history = model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test), verbose=0)
    loss_histories.append(history.history['loss'])  # Append training loss history for this configuration
    val_loss_histories.append(history.history['val_loss'])  # Append validation loss history
    
    # Step 6: Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    results.append(mse)
    
    # Collect data for DataFrame
    data.append({"Layers": layers, "MSE": mse})

# Create DataFrame from collected data
results_df = pd.DataFrame(data)

# Display results in a DataFrame
print(results_df)

# Step 7: Plot the training and validation loss curves
plt.figure(figsize=(10, 6))
for i in range(len(layer_configs)):
    plt.plot(loss_histories[i], label=f'Train Loss - {layer_configs[i]} Layers')
    plt.plot(val_loss_histories[i], label=f'Validation Loss - {layer_configs[i]} Layers', linestyle='--')
plt.title('Training and Validation Loss by Number of Hidden Layers')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Step 8: Plot the results of MSE
plt.figure(figsize=(10, 6))
plt.plot(layer_configs, results, marker='o', linestyle='-', color='b')
plt.title('Impact of Varying Number of Hidden Layers on MSE')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()
