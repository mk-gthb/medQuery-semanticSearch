import pandas as pd
import numpy as np

# Step 1: Load the Data
try:
    df = pd.read_csv('pmc_patients.csv', delimiter='\t', on_bad_lines='warn')
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Step 2: Inspect the Data
print(df.head())
print(df.info())

# Step 3: Identify and Clean Bad Lines
def clean_bad_lines(row):
    expected_fields = 10
    if len(row) != expected_fields:
        # Implement your specific cleaning logic here
        # This is a placeholder - adjust based on your data issues
        return row[:expected_fields] if len(row) > expected_fields else row + [np.nan] * (expected_fields - len(row))
    return row

# Apply Cleaning Logic
print("Cleaning bad lines...")
cleaned_rows = [clean_bad_lines(row) for _, row in df.iterrows()]
cleaned_df = pd.DataFrame(cleaned_rows, columns=df.columns)

# Step 4: Handle Missing Values
print("Handling missing values...")
# Adjust this based on which columns should allow 'Unknown' and which should be left as NaN
cleaned_df = cleaned_df.fillna('Unknown')

# Step 5: Remove Duplicates
print("Removing duplicates...")
cleaned_df = cleaned_df.drop_duplicates()

# Step 6: Standardize Data Formats
print("Standardizing data formats...")
# Convert age to numeric, coerce errors to NaN
cleaned_df['age'] = pd.to_numeric(cleaned_df['age'], errors='coerce')

# Standardize gender column - adjust based on your data
cleaned_df['gender'] = cleaned_df['gender'].str.lower()
cleaned_df['gender'] = cleaned_df['gender'].replace({'male': 'M', 'female': 'F', 'm': 'M', 'f': 'F'})

# Step 7: Handle Outliers (if necessary)
# Uncomment and adjust if you need to handle outliers
# print("Handling outliers...")
# Q1 = cleaned_df['age'].quantile(0.25)
# Q3 = cleaned_df['age'].quantile(0.75)
# IQR = Q3 - Q1
# cleaned_df = cleaned_df[(cleaned_df['age'] >= (Q1 - 1.5 * IQR)) & (cleaned_df['age'] <= (Q3 + 1.5 * IQR))]

# Step 8: Save the Cleaned Data
print("Saving cleaned data...")
cleaned_df.to_csv('cleaned_pmc_patients.csv', index=False, sep='\t')

print("Data cleaning and processing completed.")
