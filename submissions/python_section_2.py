import pandas as pd
import numpy as np
from datetime import time

def calculate_distance_matrix(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Create a DataFrame to store distances
    ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))  # Get unique IDs
    distance_matrix = pd.DataFrame(0, index=ids, columns=ids)  # Initialize the matrix with zeros

    # Populate the distance matrix
    for _, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] = row['distance']  # Make it symmetric

    # Ensure that only the diagonal has zero values while accumulating distances
    for k in ids:
        for i in ids:
            for j in ids:
                # Only update if there is a path through k
                if distance_matrix.at[i, k] > 0 and distance_matrix.at[k, j] > 0:
                    distance_matrix.at[i, j] = min(distance_matrix.at[i, j] if distance_matrix.at[i, j] != 0 else float('inf'),
                                                    distance_matrix.at[i, k] + distance_matrix.at[k, j])

    # Set the diagonal to zero
    for id_ in ids:
        distance_matrix.at[id_, id_] = 0

    return distance_matrix

if __name__ == "__main__":
    # Update with the actual link to your CSV
    file_path = "https://raw.githubusercontent.com/KhamurunnisaMD/MapUp-DA-Assessment-2024/refs/heads/main/submissions/dataset-2.csv"
    distance_matrix = calculate_distance_matrix(file_path)
    print(distance_matrix)
    
    


def unroll_distance_matrix(distance_matrix):
    # Create an empty list to hold the unrolled data
    unrolled_data = []

    # Get the IDs from the index (or columns) of the distance matrix
    ids = distance_matrix.index

    # Iterate through the distance matrix to collect non-diagonal entries
    for id_start in ids:
        for id_end in ids:
            if id_start != id_end:  # Exclude the same id_start and id_end
                distance = distance_matrix.at[id_start, id_end]
                unrolled_data.append({"id_start": id_start, "id_end": id_end, "distance": distance})

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

if __name__ == "__main__":
    # Assuming distance_matrix is already calculated
    # Example usage
    # Update with the actual link to your CSV
    file_path = r"C:\Users\Manzur\OneDrive\Documents\GitHub\MapUp-DA-Assessment-2024\submissions\dataset-2.csv"
    
    # First calculate the distance matrix
    distance_matrix = calculate_distance_matrix(file_path)
    
    # Now unroll the distance matrix
    unrolled_df = unroll_distance_matrix(distance_matrix)
    
    # Display the unrolled DataFrame
    print(unrolled_df)
    
    



def find_ids_within_ten_percentage_threshold(df, reference_id):
    # Calculate the average distance for the given reference_id
    reference_distances = df[df['id_start'] == reference_id]['distance']
    
    if reference_distances.empty:
        return []  # Return empty if reference_id is not found
    
    average_distance = reference_distances.mean()
    
    # Calculate the lower and upper thresholds
    lower_threshold = average_distance * 0.9
    upper_threshold = average_distance * 1.1
    
    # Find all ids that fall within the 10% threshold of the average distance
    within_threshold = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]
    
    # Get unique id_start values from the filtered DataFrame
    ids_within_threshold = within_threshold['id_start'].unique()
    
    # Return a sorted list of the ids
    return sorted(ids_within_threshold)

if __name__ == "__main__":
    # Example usage
    # Assume unrolled_df is the DataFrame from Question 10
    unrolled_df = pd.DataFrame({
        'id_start': [1001400, 1001400, 1001402, 1001402, 1001404],
        'id_end': [1001402, 1001404, 1001400, 1001404, 1001400],
        'distance': [9.7, 29.9, 20.2, 16.0, 11.1]
    })
    
    # Set a reference ID
    reference_id = 1001400
    
    # Find IDs within the 10% threshold
    result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
    
    # Display the result
    print(result_ids)





def calculate_toll_rate(df):
    # Define the toll rate coefficients
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type and add them as new columns
    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient
    
    return df

if __name__ == "__main__":
    # Example usage
    # Assume unrolled_df is the DataFrame from Question 10
    unrolled_df = pd.DataFrame({
        'id_start': [1001400, 1001400, 1001402, 1001402, 1001404],
        'id_end': [1001402, 1001404, 1001400, 1001404, 1001400],
        'distance': [9.7, 29.9, 20.2, 16.0, 11.1]
    })
    
    # Calculate toll rates
    toll_rates_df = calculate_toll_rate(unrolled_df)
    
    # Display the resulting DataFrame
    print(toll_rates_df)






def calculate_time_based_toll_rates(df):
    # Define time discount factors
    weekday_discount_factors = {
        (time(0, 0, 0), time(10, 0, 0)): 0.8,
        (time(10, 0, 0), time(18, 0, 0)): 1.2,
        (time(18, 0, 0), time(23, 59, 59)): 0.8
    }
    weekend_discount_factor = 0.7

    # Example start and end times for the calculations
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    start_time_value = time(0, 0, 0)  # Start of day
    end_time_value = time(23, 59, 59)  # End of day

    # Create new columns for start_day, start_time, end_day, and end_time
    df['start_day'] = days * (len(df) // len(days)) + days[:len(df) % len(days)]
    df['start_time'] = start_time_value
    df['end_day'] = df['start_day']  # Assuming same day for start and end in this case
    df['end_time'] = end_time_value

    # Calculate toll rates based on the time and day
    for index, row in df.iterrows():
        # Check if it's a weekend
        if row['start_day'] in ["Saturday", "Sunday"]:
            discount_factor = weekend_discount_factor
        else:
            # Determine the appropriate discount factor for weekdays based on time
            discount_factor = 1.0  # Default
            for time_range, factor in weekday_discount_factors.items():
                if time_range[0] <= row['start_time'] < time_range[1]:
                    discount_factor = factor
                    break
        
        # Update vehicle toll rates based on the discount factor
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            df.at[index, vehicle] *= discount_factor

    return df

if __name__ == "__main__":
    # Example usage
    # Assume toll_rates_df is the DataFrame from Question 12
    toll_rates_df = pd.DataFrame({
        'id_start': [1001400, 1001402],
        'id_end': [1001402, 1001404],
        'distance': [9.7, 20.2],
        'moto': [7.76, 16.16],
        'car': [11.64, 24.24],
        'rv': [14.55, 30.30],
        'bus': [21.34, 44.44],
        'truck': [34.92, 72.72]
    })

    # Calculate time-based toll rates
    time_based_toll_rates_df = calculate_time_based_toll_rates(toll_rates_df)

    # Display the resulting DataFrame
    print(time_based_toll_rates_df)


