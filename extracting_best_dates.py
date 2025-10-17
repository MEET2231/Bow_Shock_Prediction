import pandas as pd
import config

# Define the path to your CSV file
csv_file = config.SHOCK_DATABASE_CSV

try:
    # Read the CSV file, skipping any lines that start with a '#' comment character
    df = pd.read_csv(csv_file, comment='#')

    # Get the name of the first column (which contains the Unix timestamps)
    first_column_name = df.columns[0]
    
    # Convert the Unix seconds in the first column to a proper datetime format
    # This is crucial for time-based filtering and grouping
    df['datetime'] = pd.to_datetime(df[first_column_name], unit='s')

    # Create a new column 'month_year' with the format 'YYYY-MM' (e.g., '2017-11')
    df['month_year'] = df['datetime'].dt.strftime('%Y-%m')

    # --- Find the count specifically for November 2017 ---

    # Filter the DataFrame to include only rows where 'month_year' is '2017-11'
    november_2017_df = df[df['month_year'] == '2017-11']
    
    # The total number of crossings is simply the number of rows in the filtered DataFrame
    total_crossings = len(november_2017_df)

    # Print the result in a clear, readable format
    print(f"Found a total of {total_crossings} crossings for November 2017.")

except FileNotFoundError:
    print(f"[ERROR] The file was not found at the specified path: {csv_file}")
    print("Please make sure the file path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
