

import pandas as pd
import joblib
import numpy as np

def predict_new_project(project_data):
    """
    Loads the trained models and predicts the hours and cost for new project data.

    Args:
        project_data (dict): A dictionary containing the features of a new project.
    """
    print("--- New Project Prediction ---")

    # 1. Load the trained models
    try:
        hours_model = joblib.load('hours_model.pkl')
        cost_model = joblib.load('cost_model.pkl')
        print("Successfully loaded 'hours_model.pkl' and 'cost_model.pkl'.")
    except FileNotFoundError:
        print("Error: Model files not found. Please run train_model.py first.")
        return

    # 2. Prepare the input data
    # The input dictionary needs to be converted into a pandas DataFrame
    # with the same columns that the model was trained on.
    
    # Create a DataFrame from the input dictionary
    input_df = pd.DataFrame([project_data])

    # One-hot encode the 'project_type' to match the training data format
    # We need to add all possible project type columns and set them to 0
    # This ensures the model sees the exact same feature set.
    all_project_types = ['type_咨询服务', 'type_市场推广', 'type_硬件采购', 'type_系统集成', 'type_软件开发']
    
    # Get the one-hot encoded column name for the new project's type
    current_project_type_col = f"type_{project_data['project_type']}"

    for p_type in all_project_types:
        if p_type == current_project_type_col:
            input_df[p_type] = [True]
        else:
            input_df[p_type] = [False]
            
    # Drop the original categorical column
    input_df = input_df.drop(columns=['project_type'])

    # Ensure the column order is the same as in the training data
    # This is crucial for the model to work correctly.
    # We can get the expected feature order from one of the models.
    # Note: This assumes both models were trained on the same feature set.
    expected_features = hours_model.get_booster().feature_names
    
    # Add any missing columns with a default value (e.g., 0)
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Reorder columns to match the model's expectation
    input_df = input_df[expected_features]
    
    print("\nInput features prepared for prediction:")
    print(input_df.to_string())

    # 3. Make predictions
    predicted_hours = hours_model.predict(input_df)
    predicted_cost = cost_model.predict(input_df)

    # 4. Display the results
    print("\n--- AI Prediction Results ---")
    print(f"Predicted Reasonable Effort: {predicted_hours[0]:.2f} hours")
    print(f"Predicted Reasonable Cost: {predicted_cost[0]:.2f} (in 10k currency units)")
    print("-----------------------------")


if __name__ == '__main__':
    # --- Define a Sample New Project ---
    # You can change these values to test different scenarios.
    # This project is a 'software development' project.
    sample_project = {
        # These are the features our model expects.
        # We are simulating a "quote" for which we don't have contract/actual data yet.
        'quoted_hours': 800,
        'quoted_price': 15.0, # 150k
        'actual_contract_hours': 0, # Not known yet
        'actual_contract_price': 0, # Not known yet
        'function_points': 250,
        'interface_count': 5,
        'demand_stability_rating': 4.0,
        'team_size': 8,
        'delivery_quality_score': 0, # Not known yet
        'user_satisfaction_score': 0, # Not known yet
        'project_duration_days': 120,
        'num_technologies': 5,
        'project_type': '软件开发' # This will be one-hot encoded
    }

    predict_new_project(sample_project)

