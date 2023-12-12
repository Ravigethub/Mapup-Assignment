import pandas as pd
df=pd.read_csv("C:/Users/Ravi/Downloads/mapup/dataset-1.csv")
df1=pd.read_csv('C:/Users/Ravi/Downloads/mapup/dataset-2.csv')
car_matrix = pd.DataFrame(df)

# Create a pivot table
pivot_df = car_matrix.pivot_table(index='id_1', columns='id_2', values='car', fill_value=0)

# Set diagonal values to 0
for idx in pivot_df.index:
    pivot_df.loc[idx, idx] = 0

print(pivot_df)

#####

def get_type_count(dataframe):
    # Create a new column 'car_type' based on the values of 'car'
    dataframe['car_type'] = pd.cut(dataframe['car'], bins=[float('-inf'), 15, 25, float('inf')],
                                   labels=['low', 'medium', 'high'], right=False)
    
    # Return the updated DataFrame
    return dataframe


# Call the function to add the 'car_type' column
count = get_type_count(df)

# Print the updated DataFrame
print(count)
############


def get_bus_indexes(dataframe):
    # Calculate the mean value of the 'bus' column
    bus_mean = dataframe['bus'].mean()
    
    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = dataframe[dataframe['bus'] > 2 * bus_mean].index.tolist()
    
    # Sort the indices in ascending order
    bus_indexes.sort()
    
    # Return the sorted list of indices
    return bus_indexes


# Call the function to get the bus indices
bus_indices = get_bus_indexes(df)

# Print the sorted list of indices
print("Bus indices greater than twice the mean:", bus_indices)

###########


def filter_routes(dataframe):
    # Calculate the mean value of the 'truck' column for each route
    route_means = dataframe.groupby('route')['truck'].mean()
    
    # Filter routes where the average of 'truck' column is greater than 7
    selected_routes = route_means[route_means > 7].index.tolist()
    
    # Sort the list of selected routes
    selected_routes.sort()
    
    return selected_routes


# Call the function to get the filtered routes
filtered_routes = filter_routes(df)

# Print the sorted list of routes
print("Filtered routes:", filtered_routes)

################


def multiply_matrix(df):
    # Convert values to numeric, handling errors by coercing to NaN
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # Apply the specified logic to modify the numeric values in the DataFrame
    modified_df = df_numeric.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    
    # Round the numeric values to 1 decimal place
    modified_df = modified_df.round(1)
    
    return modified_df

# Assuming df is the DataFrame resulting from Question 1
# Call the function to get the modified DataFrame
modified_df = multiply_matrix(df)

# Print the modified DataFrame
print(modified_df)

import pandas as pd
import numpy as np

def multiply_matrix(df):
    # Convert the values to numeric
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    # Apply the multiplication logic based on the specified conditions
    modified_df = df_numeric.applymap(lambda x: x * 0.75 if pd.notna(x) and x > 20 else x * 1.25)

    # Set diagonal values to zero
    np.fill_diagonal(modified_df.values, 0)

    # Round the values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df

# Example usage with the result_matrix from Question 1
# Assuming df is the DataFrame from Question 1
modified_df = multiply_matrix(df)
print(modified_df)




########################

def verify_timestamps_completeness(df1):
    # Convert timestamp columns to datetime format with 'coerce' to handle invalid values
    df1['start_timestamp'] = pd.to_datetime(df1['startDay'] + ' ' + df1['startTime'], errors='coerce')
    df1['end_timestamp'] = pd.to_datetime(df1['endDay'] + ' ' + df1['endTime'], errors='coerce')
    
    # Create a new column for the day of the week
    df1['day_of_week'] = df1['start_timestamp'].dt.day_name()
    
    # Group by (id, id_2) and check completeness
    completeness_series = df1.groupby(['id', 'id_2']).apply(check_completeness)
    
    return completeness_series

def check_completeness(group):
    # Check if any row has NaT (Not a Timestamp) in start or end timestamp
    if group['start_timestamp'].isna().any() or group['end_timestamp'].isna().any():
        return pd.Series({'incorrect_timestamps': True})
    
    # Check if timestamps cover a full 24-hour period
    full_day_coverage = group['start_timestamp'].min().time() == pd.Timestamp('00:00:00').time() and \
                        group['end_timestamp'].max().time() == pd.Timestamp('23:59:59').time()

    # Check if timestamps span all 7 days of the week
    days_of_week = group['day_of_week'].unique()
    days_spanned = set(days_of_week) == set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    return pd.Series({'incorrect_timestamps': not (full_day_coverage and days_spanned)})


# Call the function to verify timestamps completeness
result_series = verify_timestamps_completeness(df1)

# Print the result
print(result_series)





