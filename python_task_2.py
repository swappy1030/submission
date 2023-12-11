import pandas as pd

def calculate_distance_matrix(dataset_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(r"C:\Users\DELL\Documents\GitHub\MapUp-Data-Assessment-F\datasets\dataset-3.csv")

    # Create a DataFrame to store the distance matrix
    ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))
    distance_matrix = pd.DataFrame(index=ids, columns=ids)

    # Initialize the matrix with zeros
    distance_matrix.fillna(0, inplace=True)

    # Populate the matrix with cumulative distances along known routes
    for index, row in df.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] += row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] += row['distance']

    return distance_matrix

# Example usage
# Assuming 'dataset-3.csv' is in the current working directory
dataset_path = 'dataset-3.csv'
result_matrix = calculate_distance_matrix(dataset_path)
result_matrix.to_csv('output_of_first1_dataset.csv', index=False)

# Display the result
print(result_matrix)


from itertools import product

def unroll_distance_matrix(distance_matrix):
    # Create all combinations of id_start and id_end
    id_combinations = list(product(distance_matrix.index, distance_matrix.columns))

    # Filter out combinations where id_start is equal to id_end
    id_combinations = [(id_start, id_end) for id_start, id_end in id_combinations if id_start != id_end]

    # Create the unrolled DataFrame
    unrolled_df = pd.DataFrame({
        'id_start': [id_start for id_start, _ in id_combinations],
        'id_end': [id_end for _, id_end in id_combinations],
        'distance': [distance_matrix.loc[id_start, id_end] for id_start, id_end in id_combinations]
    })

    return unrolled_df

output_file_path = 'output_of_first_dataset.csv'
df_distance_matrix = pd.read_csv(output_file_path)
result_unrolled = unroll_distance_matrix(df_distance_matrix)
result_unrolled.to_csv('output_of_second_dataset.csv', index=False)

print(result_unrolled)


def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Calculate the average distance for the reference value
    reference_avg_distance = df[df['id_start'] == reference_value]['distance'].mean()

    # Calculate the lower and upper bounds for the 10% threshold
    lower_bound = reference_avg_distance - (0.1 * reference_avg_distance)
    upper_bound = reference_avg_distance + (0.1 * reference_avg_distance)

    # Filter the DataFrame based on the 10% threshold using the 'between' method
    filtered_df = df[(df['id_start'] != reference_value) & df['distance'].between(lower_bound, upper_bound, inclusive='both')]

    # Get the unique values from the 'id_start' column and sort them
    result_ids = sorted(filtered_df['id_start'].unique())

    return result_ids

# Assuming 'output_of_second_dataset.csv' is the output file from the second dataset
output_file_path = 'output_of_second_dataset.csv'
df_second_dataset = pd.read_csv(output_file_path)

# Assuming reference_value is an integer value from 'id_start' column
reference_value = 123  # Replace with the actual reference value

result_ids_within_threshold = find_ids_within_ten_percentage_threshold(df_second_dataset, reference_value)

# Display the result
print(result_ids_within_threshold)


def calculate_toll_rate(df):
    # Add columns for toll rates
    df['moto'] = df['distance'] * 0.8
    df['car'] = df['distance'] * 1.2
    df['rv'] = df['distance'] * 1.5
    df['bus'] = df['distance'] * 2.2
    df['truck'] = df['distance'] * 3.6

    return df

# Assuming 'output_of_second_dataset.csv' is the output file from the second dataset
output_file_path = 'output_of_second_dataset.csv'
df_second_dataset = pd.read_csv(output_file_path)

# Call the function to calculate toll rates
df_with_toll_rates = calculate_toll_rate(df_second_dataset)

# Display the DataFrame with toll rates
print(df_with_toll_rates)


from datetime import time

# Assuming 'output_of_second_dataset.csv' is the output file from the second dataset
output_file_path = 'output_of_second_dataset.csv'
df_second_dataset = pd.read_csv(output_file_path)

def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Calculate the average distance for the reference value
    reference_avg_distance = df[df['id_start'] == reference_value]['distance'].mean()

    # Calculate the lower and upper bounds for the 10% threshold
    lower_bound = reference_avg_distance - (0.1 * reference_avg_distance)
    upper_bound = reference_avg_distance + (0.1 * reference_avg_distance)

    # Filter the DataFrame based on the 10% threshold
    filtered_df = df[(df['id_start'] != reference_value) & (df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]

    # Get the unique values from the 'id_start' column and sort them
    result_ids = sorted(filtered_df['id_start'].unique())

    return result_ids

def calculate_time_based_toll_rates(df):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Define time ranges
    weekday_ranges = [(time(0, 0), time(10, 0)), (time(10, 0), time(18, 0)), (time(18, 0), time(23, 59, 59))]
    weekend_ranges = [(time(0, 0), time(23, 59, 59))]

    # Apply discount factors based on time ranges
    for day_range in [weekday_ranges, weekend_ranges]:
        for start_time, end_time in day_range:
            mask = (df_copy['start_time'] >= start_time) & (df_copy['end_time'] <= end_time)
            df_copy.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= 0.8 if day_range == weekday_ranges else 0.7

    # Convert start_day and end_day to strings with day values
    df_copy['start_day'] = df_copy['start_day'].apply(lambda x: pd.to_datetime(x).day_name())
    df_copy['end_day'] = df_copy['end_day'].apply(lambda x: pd.to_datetime(x).day_name())

    # Convert start_time and end_time to datetime.time()
    df_copy['start_time'] = pd.to_datetime(df_copy['start_time']).dt.time
    df_copy['end_time'] = pd.to_datetime(df_copy['end_time']).dt.time

    return df_copy

# Assuming reference_value is an integer value from 'id_start' column
reference_value = 123  # Replace with the actual reference value

result_ids_within_threshold = find_ids_within_ten_percentage_threshold(df_second_dataset, reference_value)
print(result_ids_within_threshold)

df_with_time_based_toll_rates = calculate_time_based_toll_rates(df_second_dataset)
print(df_with_time_based_toll_rates)
