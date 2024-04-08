# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:24:02 2024

@author: aas0041
"""

import pandas as pd

# Replace 'your_file_path.xlsx' with the path to your actual file
file_path = 'C:/Users/aas0041/Desktop/590 Project/ian.xlsx'

# Reading the Excel file
df = pd.read_excel(file_path)

# Filter for rows where 'State/Province' is 'FL'
fl_df = df[df['State/Province'] == 'FL']

# Initialize a set to store unique County (US only) for FL
unique_county_fl = set(fl_df['County (US only)'])

# Convert the set elements to strings and then sort
sorted_county_fl = sorted([str(county) for county in unique_county_fl])

# Print the unique County (US only) for FL
print("Unique County Names for Florida:")
for county in sorted_county_fl:
    print(county)

# Create empty lists to store the max gust and sustained values for each county
max_gusts = []
max_sustained_values = []

# Iterate over each unique county
for county in sorted_county_fl:
    county_df = fl_df[fl_df['County (US only)'] == county].copy()  # Make a copy to avoid the SettingWithCopyWarning
    
    # Convert 'Gust (kt)' and 'Sustained (kt)' columns to numeric, ignoring errors
    county_df.loc[:, 'Gust (kt)'] = pd.to_numeric(county_df['Gust (kt)'], errors='coerce')
    county_df.loc[:, 'Sustained (kt)'] = pd.to_numeric(county_df['Sustained (kt)'], errors='coerce')
    
    # Drop rows with NaN values in 'Gust (kt)' and 'Sustained (kt)' columns
    county_df.dropna(subset=['Gust (kt)', 'Sustained (kt)'], inplace=True)
    
    # Get the max gust and sustained values for the county
    max_gust = county_df['Gust (kt)'].max()
    max_sustained = county_df['Sustained (kt)'].max()
    
    # Append the max values to the respective lists
    max_gusts.append(max_gust)
    max_sustained_values.append(max_sustained)

# Create a DataFrame to store the results
max_wind_speed_df = pd.DataFrame({
    'County': sorted_county_fl,
    'Max Gust (kt)': max_gusts,
    'Max Sustained (kt)': max_sustained_values
})

# Print the results
print(max_wind_speed_df)

# Save the results to an Excel file
max_wind_speed_df.to_excel('florida_max_wind_speed.xlsx', index=False)
