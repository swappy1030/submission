import pandas as pd

def generate_car_matrix(dataset_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(r"C:\Users\DELL\Documents\GitHub\MapUp-Data-Assessment-F\datasets\dataset-1.csv")

    # Create a pivot table using id_1 as index, id_2 as columns, and car as values
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')

    # Fill NaN values with 0
    car_matrix = car_matrix.fillna(0)

    # Set diagonal values to 0
    for col in car_matrix.columns:
        car_matrix.at[col, col] = 0

    return car_matrix

# Example usage
dataset_path = (r"C:\Users\DELL\Documents\GitHub\MapUp-Data-Assessment-F\datasets\dataset-1.csv")
result_matrix = generate_car_matrix(dataset_path)
result_matrix.to_csv('output_of_first_dataset.csv', index=False)
# Display the result
print(result_matrix)


def get_type_count(dataset_path):
    df = pd.read_csv(r"C:\Users\DELL\Documents\GitHub\MapUp-Data-Assessment-F\datasets\dataset-1.csv")

    # Create a new column 'car_type' based on conditions
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['high', 'medium', 'low'], include_lowest=True)

    # Calculate the count of occurrences for each car_type category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts


# Example usage
dataset_path = 'dataset-1.csv'
result = get_type_count(dataset_path)

# Display the result
print(result)


def get_bus_indexes(dataset_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(r"C:\Users\DELL\Documents\GitHub\MapUp-Data-Assessment-F\datasets\dataset-1.csv")

    # Calculate the mean value of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

# Example usage
dataset_path = 'dataset-1.csv'
result = get_bus_indexes(dataset_path)

# Display the result
print("Result: ", result)


def filter_routes(df):
    # Calculate the average of the 'truck' column for each 'route'
    route_averages = df.groupby('route')['truck'].mean()

    # Filter routes where the average truck value is greater than 7
    selected_routes = route_averages[route_averages > 7].index.tolist()

    # Sort the list of selected routes
    selected_routes = sorted(selected_routes)

    return selected_routes

# Example usage
# Assuming 'dataset-1.csv' is in the current working directory
dataset_path = (r'C:\Users\DELL\Documents\GitHub\MapUp-Data-Assessment-F\datasets\dataset-1.csv')
df_dataset_1 = pd.read_csv(dataset_path)

# Apply the function
result_routes = filter_routes(df_dataset_1)

# Display the result
print(result_routes)


def multiply_matrix(input_matrix):
    # Create a copy of the input matrix to avoid modifying the original DataFrame
    modified_matrix = input_matrix.copy()

    # Apply the specified logic to modify values
    modified_matrix = modified_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the modified values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

# Example usage
# Assuming 'result_matrix' is the DataFrame from Question 1
#result_matrix = generate_car_matrix('dataset-1.csv')  # Replace with the actual path
#modified_result_matrix = multiply_matrix(result_matrix)

result_matrix = pd.read_csv('output_of_first_dataset.csv')
modified_result_matrix = multiply_matrix(result_matrix)
# Display the modified result

print(modified_result_matrix)


def verify_timestamp_completeness(df):
    # Combine 'startDay' and 'startTime' columns into a single datetime column
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')

    # Combine 'endDay' and 'endTime' columns into a single datetime column
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')

    # Drop rows with NaT values (out-of-bounds timestamps)
    df = df.dropna(subset=['start_timestamp', 'end_timestamp'])

    # Calculate the difference between end and start timestamps
    df['timestamp_diff'] = df['end_timestamp'] - df['start_timestamp']

    # Check if timestamps cover a full 24-hour period and span all 7 days
    valid_timestamps = (
        (df['start_timestamp'].dt.time == pd.to_datetime('00:00:00').time()) &
        (df['end_timestamp'].dt.time == pd.to_datetime('23:59:59').time()) &
        (df['timestamp_diff'] == pd.to_timedelta('1 day')) &
        (df['start_timestamp'].dt.day_name().isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']))
    )

    # Ensure 'id' and 'id_2' are present in the DataFrame
    if 'id' not in df.columns or 'id_2' not in df.columns:
        raise ValueError("Columns 'id' and 'id_2' are required in the DataFrame.")

    # Create a boolean series with multi-index (id, id_2)
    result_series = valid_timestamps.groupby(['id', 'id_2']).all()

    return result_series

# Example usage
# Assuming 'dataset-2.csv' is in the current working directory
df_dataset_2 = pd.read_csv(r"C:\Users\DELL\Documents\GitHub\MapUp-Data-Assessment-F\datasets\dataset-1.csv")
result = verify_timestamp_completeness(df_dataset_2)

# Display the result
print(result)
