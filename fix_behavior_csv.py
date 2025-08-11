import pandas as pd
import os

# Input and output paths
input_path = 'datasets/behavioral_data.csv'  # rename your original file to this
output_path = 'datasets/clean_behavioral_data.csv'

# Read original file with tab separator
try:
    df = pd.read_csv(input_path, sep='\t')

    print("[INFO] Original columns:", df.columns.tolist())

    # Rename the label column to 'Class' (you can change this based on your file)
    # For example, if your file has 'ASD_traits' or similar
    if 'ASD_traits' in df.columns:
        df = df.rename(columns={'ASD_traits': 'Class'})
    elif 'Autism Spectrum Quotient' in df.columns:
        df = df.rename(columns={'Autism Spectrum Quotient': 'Class'})

    if 'Class' not in df.columns:
        raise ValueError("No 'Class' column found after renaming. Please check manually.")

    # Save cleaned file
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Cleaned file saved to: {output_path}")

except Exception as e:
    print("[ERROR]", str(e))
