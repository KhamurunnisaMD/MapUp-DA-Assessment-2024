def reverse_by_n(list, n):
    result = []
    for i in range(0, len(list), n):
        group = list[i:i + n]
        reversed_group = []
        for j in range(len(group)):
            reversed_group.append(group[len(group) - 1 - j])
        result.extend(reversed_group)
    return result
print(reverse_by_n([1, 2, 3, 4, 5, 6, 7, 8], 3))   
print(reverse_by_n([1, 2, 3, 4, 5], 2))           
print(reverse_by_n([10, 20, 30, 40, 50, 60, 70], 4)) 



from collections import defaultdict

def group_by_length(strings):
    length_dict = defaultdict(list)
    for string in strings:
        length = len(string)
        length_dict[length].append(string)
    return dict(sorted(length_dict.items()))
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
print(group_by_length(["one", "two", "three", "four"]))


def flatten_dict_iteratively(d):
    stack = [(d, '')] 
    result = {} 
    
    while stack:
        current_dict, parent_key = stack.pop() 
        
        for key, value in current_dict.items():
            new_key = parent_key + '.' + key if parent_key else key
            
            if isinstance(value, dict):
                stack.append((value, new_key))
            
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(item, dict):
                        stack.append((item, list_key))
                    else:
                        result[list_key] = item
            
            else:
                result[new_key] = value
    
    return result

nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened = flatten_dict_iteratively(nested_dict)
print(flattened)


from itertools import permutations

def unique_permutations(nums):
    return list(set(permutations(nums)))

input_list = [1, 1, 2]
output = unique_permutations(input_list)
print([list(p) for p in output])



import re

def find_all_dates(text):
    date_pattern = r'(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})'
    
    dates = re.findall(date_pattern, text)
    
    return dates

input_text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(input_text)
print(output)



import pandas as pd
import math

def decode_polyline(polyline_str):
    """Decode a Google Maps encoded polyline into a list of (lat, lon) tuples."""
    index = 0
    lat = 0
    lon = 0
    coordinates = []

    while index < len(polyline_str):
        b = 0
        shift = 0
        result = 0

        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break

        if result & 1:
            result = ~(result >> 1)
        else:
            result >>= 1
        lat += result

        b = 0
        shift = 0
        result = 0

        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break

        if result & 1:
            result = ~(result >> 1)
        else:
            result >>= 1
        lon += result

          coordinates.append((lat / 1e5, lon / 1e5))

    return coordinates

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on the Earth."""
    R = 6371000 
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c 

def polyline_to_dataframe(polyline_str):
    """Decode polyline and convert to DataFrame with distances."""
    coordinates = decode_polyline(polyline_str)

    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    df['distance'] = 0.0 
    for i in range(1, len(df)):
        df.at[i, 'distance'] = haversine(
            df.at[i - 1, 'latitude'], df.at[i - 1, 'longitude'],
            df.at[i, 'latitude'], df.at[i, 'longitude']
        )

    return df

polyline_str = "gfo}EtohhU"
df = polyline_to_dataframe(polyline_str)
print(df)




def rotate_matrix(matrix):
    n = len(matrix)
    rotated = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated[j][n - 1 - i] = matrix[i][j]
    
    return rotated

def transform_matrix(matrix):
    n = len(matrix)
    transformed = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(matrix[i])
            col_sum = sum(matrix[k][j] for k in range(n))
            transformed[i][j] = row_sum + col_sum - matrix[i][j]
            
    return transformed

def rotate_and_transform(matrix):
    rotated = rotate_matrix(matrix)
    transformed = transform_matrix(rotated)
    return transformed

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
final_matrix = rotate_and_transform(matrix)
print(final_matrix)



import pandas as pd

def check_time_completeness(df):
    day_mapping = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    base_date = pd.Timestamp("2024-01-01") 

    df['start_timestamp'] = pd.to_datetime(
        base_date + pd.to_timedelta(df['startDay'].map(day_mapping), unit='D') + pd.to_timedelta(df['startTime'])
    )
    
    df['end_timestamp'] = pd.to_datetime(
        base_date + pd.to_timedelta(df['endDay'].map(day_mapping), unit='D') + pd.to_timedelta(df['endTime'])
    )

    completeness_check = (df['end_timestamp'] - df['start_timestamp']) >= pd.Timedelta(days=1)
    
    return completeness_check

file_path = r"C:\Users\Manzur\OneDrive\Documents\GitHub\MapUp-DA-Assessment-2024\submissions\dataset-1.csv"
df = pd.read_csv(file_path)

incorrect_timestamps = check_time_completeness(df)

print(incorrect_timestamps)
