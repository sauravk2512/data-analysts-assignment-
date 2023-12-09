#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 


# In[43]:


# file upload 

df = pd.read_csv('dataset-1.csv')
df1 = pd.read_csv('dataset-2.csv')
df2 = pd.read_csv('dataset-3.csv')


# In[9]:


df.head


# In[10]:


df1.head


# # Python Task 1

# # Question -1 Car Matrix Generation

# In[ ]:


# Read the CSV file into a DataFrame
df = pd.read_csv(dataset_file)

# Pivot the DataFrame
result_df = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

# Set diagonal values to 0
for index in result_df.index:
    result_df.loc[index, index] = 0

return result_df

print(result_df)


# # questions -2  Car Type Count Calculation

# In[ ]:


def get_type_count(dataset_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset_file)

    # Add a new categorical column 'car_type'
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_count = dict(sorted(type_count.items()))

    return sorted_type_count

# Example usage:
file_path = 'dataset-1.csv'
result_dict = get_type_count(file_path)

# Display the resulting dictionary
print(result_dict)


# # Questions -3 Bus Count Index Retrieval

# In[31]:


import pandas as pd

def get_bus_indexes(dataset_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset_file)

    # Calculate the mean value of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

# Example usage:
file_path = 'dataset-1.csv'
result_list = get_bus_indexes(file_path)

# Display the resulting list of indices
print(result_list)



# # Questions -4 Route Filtering

# In[33]:


def filter_routes(dataset_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset_file)

    # Group by 'route' and calculate the average of 'truck' for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average 'truck' value is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes
    selected_routes.sort()

    return selected_routes

# Example usage:
file_path = 'dataset-1.csv'
result_list = filter_routes(file_path)

# Display the resulting sorted list of routes
print(result_list)


# # Questions -5 Matrix Value Modification

# In[40]:


def multiply_matrix(result_df):
    # Create a copy of the input DataFrame to avoid modifying the original
    modified_df = result_df.copy()

    # Apply the specified logic to modify values
    modified_df = modified_df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df

# Example usage (assuming result_df is the DataFrame from Question 1):
# result_df = generate_car_matrix('path/to/dataset-1.csv')
# modified_result_df = multiply_matrix(result_df)

# Display the modified DataFrame
modified_result_df = multiply_matrix(result_df)
print(modified_result_df)


# In[ ]:





# In[ ]:





# # Python Task 2  

# In[62]:


df2.tail


# # questions -1 Distance Matrix Calculation

# In[91]:


import pandas as pd

def calculate_distance_matrix(dataset_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset_file)

    # Create a DataFrame with cumulative distances along known routes
    distance_matrix = pd.pivot_table(df, values='distance', index='id_start', columns='id_end', aggfunc='sum', fill_value=0)

    # Make the matrix symmetric
    distance_matrix = distance_matrix + distance_matrix.T

    # Set diagonal values to 0
    distance_matrix.values[[range(len(distance_matrix))]*2] = 0

    return distance_matrix

# Example usage:
file_path = 'dataset-3.csv'
distance_matrix_df = calculate_distance_matrix(file_path)

# Display the resulting DataFrame
print(distance_matrix_df)


# #  questions -2 Unroll Distance Matrix

# In[ ]:


import pandas as pd

def unroll_distance_matrix(input_df):
    # Create a copy of the input DataFrame to avoid modifying the original
    df = input_df.copy()

    # Melt the DataFrame to convert it to long format
    melted_df = pd.melt(df, id_vars=['id'], var_name='id_end', value_name='distance')

    # Rename the columns to match the expected output
    melted_df.rename(columns={'id': 'id_start'}, inplace=True)

    # Filter out rows where id_start is the same as id_end
    result_df = melted_df[melted_df['id_start'] != melted_df['id_end']]

    # Sort the columns for consistency
    result_df = result_df[['id_start', 'id_end', 'distance']]

    return result_df

# Example usage:
file_path = 'dataset-3.csv'
distance_matrix_df = generate_car_matrix(file_path)
unrolled_df = unroll_distance_matrix(distance_matrix_df)

# Display the resulting DataFrame
print(unrolled_df)


# # questions -3 Finding IDs within Percentage Threshold

# In[ ]:


import pandas as pd

def find_ids_within_ten_percentage_threshold(input_df, reference_value):
    # Create a copy of the input DataFrame to avoid modifying the original
    df = input_df.copy()

    # Filter rows with the specified reference value
    reference_rows = df[df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    reference_avg_distance = reference_rows['distance'].mean()

    # Calculate the 10% threshold range
    threshold_lower = reference_avg_distance - (reference_avg_distance * 0.1)
    threshold_upper = reference_avg_distance + (reference_avg_distance * 0.1)

    # Filter rows within the 10% threshold range
    within_threshold_df = df[(df['distance'] >= threshold_lower) & (df['distance'] <= threshold_upper)]

    # Get unique values from the 'id_start' column and sort them
    result_list = sorted(within_threshold_df['id_start'].unique())

    return result_list

# Example usage:
file_path = 'dataset-3.csv'
distance_matrix_df = calculate_distance_matrix(file_path)
reference_value = 1  # Replace with the desired reference value
result_list = find_ids_within_ten_percentage_threshold(distance_matrix_df, reference_value)

# Display the resulting sorted list
print(result_list)


# # Questions -4  Calculate Toll Rate

# In[85]:


import pandas as pd

def calculate_toll_rate(input_df):
    # Create a copy of the input DataFrame to avoid modifying the original
    df = input_df.copy()

    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        column_name = f'{vehicle_type}_rate'
        df[column_name] = df['distance'] * rate_coefficient

    return df

# Example usage (assuming result_df is the DataFrame from Question 2):
# result_df = generate_car_matrix('path/to/dataset-1.csv')
# result_df = calculate_toll_rate(result_df)

# Display the resulting DataFrame with toll rates
print(result_df)


# # questions -5 Calculate Time-Based Toll Rates

# In[86]:


import pandas as pd
from datetime import datetime, time

def calculate_time_based_toll_rates(input_df):
    # Create a copy of the input DataFrame to avoid modifying the original
    df = input_df.copy()

    # Define time ranges and discount factors
    weekday_time_ranges = [(time(0, 0, 0), time(10, 0, 0)),
                           (time(10, 0, 0), time(18, 0, 0)),
                           (time(18, 0, 0), time(23, 59, 59))]
    weekend_time_range = (time(0, 0, 0), time(23, 59, 59))
    
    weekday_discount_factors = [0.8, 1.2, 0.8]
    weekend_discount_factor = 0.7

    # Create new columns for start_day, start_time, end_day, and end_time
    df['start_day'] = df['timestamp'].dt.strftime('%A')
    df['end_day'] = df['start_day']
    df['start_time'] = df['timestamp'].dt.time
    df['end_time'] = df['start_time']

    # Apply discount factors based on time ranges
    for i, (start_time, end_time) in enumerate(weekday_time_ranges):
        weekday_condition = (df['timestamp'].dt.time >= start_time) & (df['timestamp'].dt.time < end_time) & (df['start_day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))
        df.loc[weekday_condition, df.columns[5:]] *= weekday_discount_factors[i]

    weekend_condition = df['start_day'].isin(['Saturday', 'Sunday'])
    df.loc[weekend_condition, df.columns[5:]] *= weekend_discount_factor

    return df

# Example usage (assuming result_df is the DataFrame from Question 3):
# result_df = calculate_distance_matrix('path/to/dataset-3.csv')
# result_df = calculate_time_based_toll_rates(result_df)

# Display the resulting DataFrame with time-based toll rates
print(result_df)


# In[ ]:




