


import pandas as pd
df=pd.read_csv("C:/Users/Ravi/Downloads/mapup/dataset-3.csv")
import networkx as nx
import numpy as np

def calculate_distance_matrix(df):
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges with distances from the DataFrame
    for index, row in df.iterrows():
        G.add_edge(row['id_start'], row['id_end'], weight=row['distance'])
        G.add_edge(row['id_end'], row['id_start'], weight=row['distance'])  # Add reverse edge for bidirectionality

    # Calculate all-pairs shortest paths
    all_pairs_shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G))

    # Create a DataFrame for the distance matrix
    distance_matrix = pd.DataFrame(all_pairs_shortest_paths).fillna(0)

    # Set diagonal values to zero using numpy
    np.fill_diagonal(distance_matrix.values, 0)

    return distance_matrix

# Example usage with the provided DataFrame df
# Replace 'dataset-3.csv' with the actual path to your CSV file
result_matrix = calculate_distance_matrix(df[['id_start', 'id_end', 'distance']])
print(result_matrix)




def unroll_distance_matrix(distance_matrix_df):
    unrolled_data = []

    for id_start in distance_matrix_df.index:
        for id_end in distance_matrix_df.columns:
            if id_start != id_end:
                distance = distance_matrix_df.loc[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    result_df = pd.DataFrame(unrolled_data)
    return result_df

# Example usage with the result_distance_matrix
# Assuming result_distance_matrix is the DataFrame from Question 1
result_unrolled_distance = unroll_distance_matrix(result_matrix)
print(result_unrolled_distance)




def find_ids_within_ten_percentage_threshold(dataframe, reference_value):
    reference_rows = dataframe[dataframe['id_start'] == reference_value]
    reference_average_distance = reference_rows['distance'].mean()
    lower_bound = reference_average_distance - 0.1 * reference_average_distance
    upper_bound = reference_average_distance + 0.1 * reference_average_distance
    within_threshold_ids = dataframe[(dataframe['distance'] >= lower_bound) & (dataframe['distance'] <= upper_bound)]
    sorted_ids_within_threshold = sorted(within_threshold_ids['id_start'].unique())
    return sorted_ids_within_threshold
reference_value = 1
result_within_threshold = find_ids_within_ten_percentage_threshold(result_unrolled_distance, reference_value)
print(result_within_threshold)


def calculate_toll_rate(input_dataframe):
    result_dataframe = input_dataframe.copy()
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        result_dataframe[vehicle_type] = result_dataframe['distance'] * rate_coefficient
        return result_dataframe
result_with_toll_rate = calculate_toll_rate(result_unrolled_distance)
print(result_with_toll_rate)


import pandas as pd
import datetime

def calculate_time_based_toll_rates(input_dataframe):
    result_dataframe = input_dataframe.copy()
    time_ranges = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0), 0.8),
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0), 1.2),
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59), 0.8)
        ]
    weekend_discount_factor = 0.7
    for id_start in result_dataframe['id_start'].unique():
        for id_end in result_dataframe['id_end'].unique():
            for start_time, end_time, discount_factor in time_ranges
            condition = (result_dataframe['id_start'] == id_start) & (result_dataframe['id_end'] == id_end)
            result_dataframe.loc[condition, 'start_day'] = 'Monday'
            result_dataframe.loc[condition, 'end_day'] = 'Sunday'
            result_dataframe.loc[condition, 'start_time'] = start_time
            result_dataframe.loc[condition, 'end_time'] = end_time
            result_dataframe.loc[condition, ['moto', 'car', 'rv', 'bus', 'truck']] *= discount_factor
            condition_weekend = (result_dataframe['id_start'] == id_start) & (result_dataframe['id_end'] == id_end)
            result_dataframe.loc[condition_weekend, ['moto', 'car', 'rv', 'bus', 'truck']] *= weekend_discount_factor
            return result_dataframe

result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_with_toll_rate)
print(result_with_time_based_toll_rates)

result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_with_toll_rate)
print(result_with_time_based_toll_rates)
