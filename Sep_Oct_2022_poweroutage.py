# -*- coding: utf-8 -*-
"""
Created on Thu May  2 08:49:19 2024

@author: aas0041
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Define the file name
file_name = 'C:/Users/aas0041/Desktop/590 Project/poweroutage_data.csv'

# Initialize lists to store data for each column
CountyName_list = []
CustomersTracked_list = []
MaxCustomersOut_list = []
CustomerHoursOutTotal_list = []
CustomerHoursTrackedTotal_list = []
RecordDate_list = []

# Define the list of CountyNames to filter with only first letter capitalized
county_names_to_filter = [
    "Alachua", "Brevard", "Broward", "Charlotte", "Collier", "Desoto", "Dixie", "Duval", "Flagler",
    "Glades", "Hardee", "Hendry", "Hernando", "Highlands", "Hillsborough", "Indian River", "Lake",
    "Lee", "Manatee", "Martin", "Miami-dade", "Monroe", "Nan", "Okeechobee", "Orange", "Osceola",
    "Palm Beach", "Pasco", "Pinellas", "Polk", "Sarasota", "Seminole", "St. Johns", "St. Lucie",
    "Volusia"
]

# Open the CSV file
with open(file_name, 'r') as csv_file:
    # Read each row in the file
    for idx, csv_row in enumerate(csv_file):
        # Skip the first row (header)
        if idx == 0:
            continue

        # Split the row by commas
        data = csv_row.strip().split(',')

        # Filter data to only include entries with StateName = "Florida" and CountyName in the specified list
        if data[1].strip('"') == "Florida" and data[2].strip('"') in county_names_to_filter:
            # Append each value to the corresponding list
            CountyName_list.append(data[2].strip('"'))
            CustomersTracked_list.append(float(data[4].strip('"')))
            MaxCustomersOut = float(data[5].strip('"'))
            if MaxCustomersOut >= 0:  # Exclude data below 1
                MaxCustomersOut_list.append(MaxCustomersOut)
            else:
                MaxCustomersOut_list.append(None)

            # Check if CustomerHoursOutTotal is empty string
            if data[6].strip('"') != '':
                CustomerHoursOutTotal = float(data[6].strip('"'))
                if CustomerHoursOutTotal >= 0:  # Exclude data below 1
                    CustomerHoursOutTotal_list.append(CustomerHoursOutTotal)
                else:
                    CustomerHoursOutTotal_list.append(None)
            else:
                CustomerHoursOutTotal_list.append(None)

            # Append CustomerHoursTrackedTotal to the list if not empty
            if data[7].strip('"'):
                CustomerHoursTrackedTotal_list.append(float(data[7].strip('"')))
            else:
                CustomerHoursTrackedTotal_list.append(None)

            RecordDate_list.append(datetime.strptime(data[8].strip('"'), '%Y-%m-%d'))

# Filter data for the specified date range
start_date = datetime(2022, 9, 27)
end_date = datetime(2022, 10, 27)

# Initialize a dictionary to store the sum of 'MaxCustomersOut', 'CustomersTracked', and 'CustomerHoursOutTotal' for each unique 'RecordDate' and 'CountyName' combination
sum_customers_out_dict = {}

# Iterate over the filtered data within the specified date range
for i in range(len(RecordDate_list)):
    record_date = RecordDate_list[i]
    county_name = CountyName_list[i]
    max_customers_out = MaxCustomersOut_list[i]
    customers_tracked = CustomersTracked_list[i]
    customer_hours_out_total = CustomerHoursOutTotal_list[i]
    customer_hours_tracked_total = CustomerHoursTrackedTotal_list[i]

    # Check if the record date is within the specified range
    if start_date <= record_date <= end_date:
        # If the max_customers_out is None, skip this iteration
        if max_customers_out is None:
            continue
        
        # Check if the 'CountyName' and 'RecordDate' combination is already in the dictionary
        if (county_name, record_date) not in sum_customers_out_dict:
            # Add the 'CountyName' and 'RecordDate' combination to the dictionary
            sum_customers_out_dict[(county_name, record_date)] = (max_customers_out, customers_tracked, customer_hours_out_total, customer_hours_tracked_total)
        else:
            # Add the values to the existing sum in the dictionary
            current_sum = sum_customers_out_dict[(county_name, record_date)]
            updated_sum = (current_sum[0] + max_customers_out, current_sum[1] + customers_tracked, current_sum[2] + customer_hours_out_total, current_sum[3] + customer_hours_tracked_total)
            sum_customers_out_dict[(county_name, record_date)] = updated_sum


# Convert the dictionary into a DataFrame
df = pd.DataFrame(sum_customers_out_dict.items(), columns=['CountyName_RecordDate', 'SumValues'])

# Split the 'CountyName_RecordDate' column into separate 'CountyName' and 'RecordDate' columns
df[['CountyName', 'RecordDate']] = df['CountyName_RecordDate'].apply(lambda x: pd.Series(x))

# Split the 'SumValues' column into separate columns
df[['MaxCustomersOut', 'CustomersTracked', 'CustomerHoursOutTotal', 'CustomerHoursTrackedTotal']] = df['SumValues'].apply(lambda x: pd.Series(x))

# Drop the unnecessary columns
df.drop(columns=['CountyName_RecordDate', 'SumValues'], inplace=True)



# Select the maximum summed values for 'CustomersTracked' and 'CustomerHoursTrackedTotal' by 'CountyName'
max_customers_tracked = df.groupby('CountyName')['CustomersTracked'].max().reset_index()
max_customer_hours_tracked_total = df.groupby('CountyName')['CustomerHoursTrackedTotal'].max().reset_index()

# Merge these maximum values back to the DataFrame
df = df.merge(max_customers_tracked, on='CountyName', suffixes=('', '_Max'))
df = df.merge(max_customer_hours_tracked_total, on='CountyName', suffixes=('', '_Max'))

# Update the DataFrame to use the maximum values
df['CustomersTracked'] = df['CustomersTracked_Max']
df['CustomerHoursTrackedTotal'] = df['CustomerHoursTrackedTotal_Max']

# Drop temporary columns
df.drop(columns=['CustomersTracked_Max', 'CustomerHoursTrackedTotal_Max'], inplace=True)


# Calculate 'Outage Density'
df['Outage Density'] = (df['MaxCustomersOut'] * df['CustomerHoursOutTotal'])*100 / (df['CustomersTracked'] * df['CustomerHoursTrackedTotal'])






# INDIVIDUAL COUNTY PLOTS

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Plot 'Outage Density' against 'RecordDate' for each county
for county in df['CountyName'].unique():
    county_data = df[df['CountyName'] == county]
    plt.figure(figsize=(5, 3))  # Adjust figure size if needed
    plt.plot(county_data['RecordDate'], county_data['Outage Density'], label=county)
    plt.xlabel('Record Date')
    plt.ylabel('Outage Density')
    plt.title(f'Outage Density vs. Record Date for {county}')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(DateFormatter('%m-%d'))  # Set date format
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()




# COMBINED COUNTY PLOTS

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Get unique county names
counties = df['CountyName'].unique()

# Calculate the number of rows and columns
num_counties = len(counties)
num_rows = 8  # Number of rows
num_cols = (num_counties + num_rows - 1) // num_rows  # Number of columns

# Create subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Plot 'Outage Density' against 'RecordDate' for each county
for i, county in enumerate(counties):
    row = i // num_cols
    col = i % num_cols
    county_data = df[df['CountyName'] == county]
    axs[row, col].plot(county_data['RecordDate'], county_data['Outage Density'], label=county)
    axs[row, col].set_ylabel('Outage Density', fontsize=6)  # Adjust axis label font size
    axs[row, col].set_title(f'Outage Density vs. Record Date for {county}', fontsize=8)  # Adjust title font size
    axs[row, col].legend(fontsize=6)  # Adjust legend font size
    axs[row, col].grid(True)
    axs[row, col].xaxis.set_major_formatter(DateFormatter('%m-%d'))  # Set date format
    axs[row, col].tick_params(axis='x', rotation=90, labelsize=6)  # Rotate x-axis labels and adjust font size
    axs[row, col].tick_params(axis='y', labelsize=6)  # Adjust y-axis font size

# Set common x-axis label
fig.text(0.5, 0.04, 'Record Date', ha='center', fontsize=12)  # Adjust x-axis label font size

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()








# Remove hours, minutes, and seconds from the 'RecordDate' column
df['RecordDate'] = df['RecordDate'].dt.date

# Save the DataFrame to a spreadsheet file (e.g., Excel)
output_file = 'Sep_Oct_2022_poweroutage.xlsx'
df.to_excel(output_file, index=False)

# Print the filtered data
print('CountyName\tRecordDate\tSumMaxCustomersOut\tSumCustomersTracked\tSumCustomerHoursOutTotal\tCustomerHoursTrackedTotal\tOutage Density')
for _, row in df.iterrows():
    print(f"{row['CountyName']}\t{row['RecordDate']}\t{row['MaxCustomersOut']}\t{row['CustomersTracked']}\t{row['CustomerHoursOutTotal']}\t{row['CustomerHoursTrackedTotal']}\t{row['Outage Density']}")
