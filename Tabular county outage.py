# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:39:02 2024

@author: aas0041
"""

import pandas as pd
from collections import defaultdict

# Define the file name
file_name = 'C:/Users/aas0041/Desktop/590 Project/Ian.csv'

# Initialize a set to store unique county names for Florida
unique_county_names_florida = set()

# Open the CSV file
with open(file_name, 'r') as csv_file:
    # Read each row in the file
    for idx, csv_row in enumerate(csv_file):
        # Split the row by commas
        data = csv_row.strip().split(',')
        
        # Only proceed if the StateName is "Florida"
        if data[1].strip('"') == "Florida":
            # Add the county name to the set (duplicates are automatically ignored)
            unique_county_names_florida.add(data[2].strip('"'))

# Convert the set to a list and sort it
sorted_unique_county_names_florida = sorted(list(unique_county_names_florida))

# Initialize dictionaries to store sums for each 'CountyName' in Florida
customers_tracked_sum = defaultdict(float)
max_customers_out_sum = defaultdict(float)
customer_hours_out_total_sum = defaultdict(float)
customer_hours_tracked_total_sum = defaultdict(float)

# Open the CSV file again
with open(file_name, 'r') as csv_file:
    # Skip the header row
    next(csv_file)
    
    # Read each row in the file
    for csv_row in csv_file:
        # Split the row by commas
        data = csv_row.strip().split(',')
        
        # Extract the 'CountyName' and other values from the row
        state_name = data[1].strip('"')
        county_name = data[2].strip('"')
        
        # Check if the state is "Florida" before updating the sums
        if state_name == "Florida":
            # Check if the value is not empty before converting to float
            if data[4].strip('"'):
                customers_tracked = float(data[4].strip('"'))
                customers_tracked_sum[county_name] += customers_tracked
            
            if data[5].strip('"'):
                max_customers_out = float(data[5].strip('"'))
                max_customers_out_sum[county_name] += max_customers_out
            
            if data[6].strip('"'):
                customer_hours_out_total = float(data[6].strip('"'))
                customer_hours_out_total_sum[county_name] += customer_hours_out_total
            
            if data[7].strip('"'):
                customer_hours_tracked_total = float(data[7].strip('"'))
                customer_hours_tracked_total_sum[county_name] += customer_hours_tracked_total

# Create a DataFrame to store the results
florida_max_outage_df = pd.DataFrame({
    'CountyName': sorted_unique_county_names_florida,
    'CustomersTracked': [customers_tracked_sum[county] for county in sorted_unique_county_names_florida],
    'MaxCustomersOut': [max_customers_out_sum[county] for county in sorted_unique_county_names_florida],
    'CustomerHoursOutTotal': [customer_hours_out_total_sum[county] for county in sorted_unique_county_names_florida],
    'CustomerHoursTrackedTotal': [customer_hours_tracked_total_sum[county] for county in sorted_unique_county_names_florida]
})

# Print the results
print(florida_max_outage_df)

# Save the results to an Excel file
output_file = 'florida_max_outage.xlsx'
florida_max_outage_df.to_excel(output_file, index=False)
