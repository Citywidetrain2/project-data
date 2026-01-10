import pandas as pd
import os

# Define paths
INPUT_FILE = 'DATA_DISFRAZADA_SUPERMERCADO.xlsx'
OUTPUT_FILE = 'cleaned_sales.csv'

def load_and_clean_data():
    print("Loading data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_excel(INPUT_FILE)
    
    # 1. Basic Inspection
    print("\n--- Initial Info ---")
    print(df.info())
    print("\n--- First 5 rows ---")
    print(df.head())

    # 2. Handle Duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    final_rows = len(df)
    print(f"\nDuplicates removed: {initial_rows - final_rows}")

    # 3. Handle Missing Values
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    # Note: If there are missing values, we would handle them here. 
    # For now, we just report them. If critical columns are missing, we might drop rows.
    
    # 4. Convert Data Types
    # 'Date' column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        print("\nConverted 'Date' column to datetime objects.")
    
    # 5. Save Cleaned Data
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nCleaned data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    load_and_clean_data()
