

import pandas as pd
import numpy as np
import json

def load_data(data_path="."):
    """Loads all necessary CSV files from the specified path."""
    print("Loading data from CSV files...")
    try:
        projects_df = pd.read_csv(f"{data_path}/projects.csv")
        vendors_df = pd.read_csv(f"{data_path}/vendors.csv")
        quotes_df = pd.read_csv(f"{data_path}/quotes.csv")
        actuals_df = pd.read_csv(f"{data_path}/project_actuals.csv")
        print("All CSV files loaded successfully.")
        return projects_df, vendors_df, quotes_df, actuals_df
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Make sure all CSV files are in the correct directory.")
        return None, None, None, None

def merge_data(projects_df, quotes_df, actuals_df):
    """Merges the individual dataframes into a single modeling dataframe."""
    print("Merging dataframes...")
    
    # We only care about completed projects with actuals for training
    # and we need to select the "winning" quote for each project
    
    # 1. Filter for quotes that have a corresponding actuals record
    completed_project_ids = actuals_df['project_id'].unique()
    
    # 2. From the quotes, only keep those for completed projects
    training_quotes_df = quotes_df[quotes_df['project_id'].isin(completed_project_ids)].copy()
    
    # 3. Merge project information into the quotes
    merged_df = pd.merge(training_quotes_df, projects_df, left_on='project_id', right_on='id', suffixes=('_quote', '_project'))
    
    # 4. Merge actuals information
    # Since actuals are 1-to-1 with projects, we can merge directly
    final_df = pd.merge(merged_df, actuals_df, on='project_id', suffixes=('', '_actual'))
    
    # Clean up redundant ID columns
    final_df = final_df.drop(columns=['id_project'])
    final_df.rename(columns={'id_quote': 'quote_id', 'id': 'actual_id'}, inplace=True)
    
    print(f"Data merged. Shape of final dataframe: {final_df.shape}")
    return final_df

def feature_engineering(df):
    """Creates new features and cleans the merged dataframe."""
    print("Performing feature engineering...")
    
    # --- Date Features ---
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['contract_date'] = pd.to_datetime(df['contract_date'])
    
    # Calculate project duration in days
    df['project_duration_days'] = (df['end_date'] - df['start_date']).dt.days
    
    # --- JSON Feature ---
    # Safely parse the JSON string in 'technology_stack'
    def parse_tech_stack(stack_str):
        try:
            return json.loads(stack_str)
        except (json.JSONDecodeError, TypeError):
            return []
            
    df['tech_stack_list'] = df['technology_stack'].apply(parse_tech_stack)
    df['num_technologies'] = df['tech_stack_list'].apply(len)

    # --- Categorical Feature Cleaning ---
    # One-hot encode project_type for simplicity in this first pass
    df = pd.get_dummies(df, columns=['project_type'], prefix='type')
    
    # --- Target Variable Definition ---
    # Our goal is to predict the actuals, so we define them as targets
    df.rename(columns={
        'actual_effort_hours': 'target_hours',
        'actual_final_cost': 'target_cost'
    }, inplace=True)
    
    # --- Clean up and select final columns ---
    # Drop original complex columns that have been engineered
    df = df.drop(columns=['technology_stack', 'tech_stack_list', 'start_date', 'end_date', 'contract_date', 'payment_terms', 'description', 'name'])
    
    # Handle potential missing values (a simple strategy for now)
    # For numeric columns, fill with the median
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled NaNs in '{col}' with median value: {median_val}")

    print("Feature engineering complete.")
    return df

if __name__ == '__main__':
    # Define the path to the data files
    DATA_PATH = "." 
    
    # Run the pipeline
    projects, vendors, quotes, actuals = load_data(DATA_PATH)
    
    if projects is not None:
        # Merge the data
        merged_data = merge_data(projects, quotes, actuals)
        
        # Engineer features
        processed_data = feature_engineering(merged_data)
        
        # Save the processed data
        output_path = f"{DATA_PATH}/processed_modeling_data.csv"
        processed_data.to_csv(output_path, index=False)
        
        print(f"\nData processing pipeline complete!")
        print(f"Processed data saved to: {output_path}")
        print("Final data columns:", processed_data.columns.tolist())
        print("Sample of processed data:")
        print(processed_data.head())

