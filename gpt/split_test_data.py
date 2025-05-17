import pandas as pd
import math
import os

def split_csv(input_file, num_parts=10):
    # Create output directory if it doesn't exist
    output_dir = 'split_test_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the CSV file
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Calculate the size of each part
    total_rows = len(df)
    rows_per_part = math.ceil(total_rows / num_parts)
    
    print(f"Total rows: {total_rows}")
    print(f"Rows per part: {rows_per_part}")
    
    # Split and save each part
    for i in range(num_parts):
        start_idx = i * rows_per_part
        end_idx = min((i + 1) * rows_per_part, total_rows)
        
        # Get the subset of data
        part_df = df.iloc[start_idx:end_idx]
        
        # Create output filename
        output_file = os.path.join(output_dir, f'test_part_{i+1}.csv')
        
        # Save to CSV
        part_df.to_csv(output_file, index=False)
        print(f"Created {output_file} with {len(part_df)} rows")

def main():
    input_file = 'test.csv'
    split_csv(input_file)
    print("\nSplitting complete!")

if __name__ == "__main__":
    main() 