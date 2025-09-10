import pandas as pd
import numpy as np
from tqdm import tqdm

def detect_and_handle_outliers(input_file, output_file, chunk_size=10000):
    # Initialize an empty DataFrame to store processed chunks
    processed_chunks = []
    
    # First pass: calculate global statistics for each column
    print("Calculating column statistics...")
    numeric_cols = []
    
    # Read the first chunk to get column names and identify numeric columns
    first_chunk = pd.read_csv(input_file, nrows=5)
    numeric_cols = first_chunk.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("No numeric columns found in the dataset.")
        return
        
    print(f"Processing numeric columns: {', '.join(numeric_cols)}")
    
    # Initialize a reader for the CSV file
    reader = pd.read_csv(input_file, chunksize=chunk_size, dtype=str)
    
    # Process each chunk
    for i, chunk in enumerate(tqdm(reader, desc="Processing chunks")):
        # Convert numeric columns to float
        for col in numeric_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        
        # Handle outliers in each numeric column
        for col in numeric_cols:
            if col in chunk.columns:
                # Calculate IQR
                Q1 = chunk[col].quantile(0.25)
                Q3 = chunk[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds (1.5 * IQR is a common threshold)
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outlier_mask = (chunk[col] < lower_bound) | (chunk[col] > upper_bound) | chunk[col].isna()
                
                # Replace outliers with NaN first
                chunk.loc[outlier_mask, col] = np.nan
                
                # Forward fill and backward fill to get surrounding values
                chunk[col] = chunk[col].fillna(method='ffill')
                chunk[col] = chunk[col].fillna(method='bfill')
                
                # If there are still NaNs (e.g., at the start or end), fill with column mean
                if chunk[col].isna().any():
                    chunk[col] = chunk[col].fillna(chunk[col].mean())
        
        # Store the processed chunk
        processed_chunks.append(chunk)
    
    # Combine all processed chunks
    print("Combining processed chunks...")
    cleaned_data = pd.concat(processed_chunks, ignore_index=True)
    
    # Save the cleaned data to a new file
    print(f"Saving cleaned data to {output_file}...")
    cleaned_data.to_csv(output_file, index=False)
    print("Data cleaning complete!")

if __name__ == "__main__":
    input_file = "data_full.csv"
    output_file = "data_cleaned.csv"
    detect_and_handle_outliers(input_file, output_file)
