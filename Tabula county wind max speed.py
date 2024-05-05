# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:22:32 2024

@author: aas0041
"""

import pandas as pd

# Path to the Excel file
file_path = 'C:/Users/aas0041/Desktop/590 Project/windspeed_data.xlsx'

# Reading the Excel file
df = pd.read_excel(file_path)

# Creating a copy of the DataFrame for operations to avoid SettingWithCopyWarning when modifying
fl_df = df[df['State/Province'] == 'FL'].copy()

# Convert all entries in 'County (US only)' column to strings and uppercase
fl_df['County (US only)'] = fl_df['County (US only)'].astype(str).str.upper()

# Extracting and deduplicating the 'County (US only)' column
unique_county_fl = sorted(fl_df['County (US only)'].dropna().unique())
print("Unique County Names for Florida:")
for county in unique_county_fl:
    print(county)

# Initialize lists to store the max gust and sustained wind speed values for each county
max_gusts = []
max_sustained_values = []

# Process each unique county
for county in unique_county_fl:
    # Filter data using a boolean mask and .copy() to ensure we are working with a separate DataFrame
    mask = fl_df['County (US only)'] == county
    county_df = fl_df.loc[mask].copy()

    print(f"Processing data for county: {county}, Entries: {county_df.shape[0]}")
    # Debug: Print count of data points per county
    print(f"Data points for {county}: {county_df.shape[0]}")

    # Safely convert data columns to numeric values, handling non-numeric gracefully
    county_df.loc[:, 'Gust (kt)'] = pd.to_numeric(county_df['Gust (kt)'].astype(str).str.replace('*', ''), errors='coerce')
    county_df.loc[:, 'Sustained (kt)'] = pd.to_numeric(county_df['Sustained (kt)'].astype(str).str.replace('*', ''), errors='coerce')

    # Calculate the maximum gust and sustained wind speeds
    max_gust = county_df['Gust (kt)'].max()
    max_sustained = county_df['Sustained (kt)'].max()

    # Append results to lists
    max_gusts.append(max_gust)
    max_sustained_values.append(max_sustained)

# Create a DataFrame to summarize the results
max_wind_speed_df = pd.DataFrame({
    'County': unique_county_fl,
    'Max Sustained (kt)': max_sustained_values,
    'Max Gust (kt)': max_gusts
})

# Print the results
print(max_wind_speed_df)

# Save the results to an Excel file
max_wind_speed_df.to_excel('florida_max_wind_speed.xlsx', index=False)

