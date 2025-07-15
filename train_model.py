

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np
import matplotlib.pyplot as plt

def train():
    """
    Loads the processed data, trains two separate XGBoost models for hours and cost,
    evaluates them, and saves the trained models to disk.
    """
    print("Starting model training process...")

    # 1. Load the processed data
    try:
        data_path = "./processed_modeling_data.csv"
        df = pd.read_csv(data_path)
        print(f"Successfully loaded processed data from {data_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {data_path}.")
        print("Please run the data_processor.py script first.")
        return

    # 2. Define features (X) and targets (y)
    # We will predict 'target_hours' and 'target_cost'
    # All other columns will be used as features, except for identifiers
    
    # Ensure target columns exist
    if 'target_hours' not in df.columns or 'target_cost' not in df.columns:
        print("Error: Target columns ('target_hours', 'target_cost') not found in the data.")
        return

    # Drop non-feature columns
    features = df.drop(columns=['target_hours', 'target_cost', 'quote_id', 'project_id', 'vendor_id', 'actual_id', 'status'])
    
    # Ensure all feature columns are numeric. This is a safeguard.
    # XGBoost requires numeric inputs.
    features = features.select_dtypes(include=np.number)
    
    target_hours = df['target_hours']
    target_cost = df['target_cost']

    print(f"Features for training: {features.columns.tolist()}")

    # --- Train Model for Target Hours ---
    print("\n--- Training model for Target Hours ---")
    
    # 3. Split data for the hours model
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        features, target_hours, test_size=0.2, random_state=42
    )

    # 4. Initialize and train the XGBoost Regressor for hours
    hours_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1 # Use all available CPU cores
    )
    
    print("Training hours model...")
    hours_model.fit(X_train_h, y_train_h)

    # 5. Evaluate the hours model
    print("Evaluating hours model...")
    predictions_h = hours_model.predict(X_test_h)
    mae_h = mean_absolute_error(y_test_h, predictions_h)
    print(f"Mean Absolute Error (Hours Model): {mae_h:.2f} hours")

    # 6. Save the hours model
    joblib.dump(hours_model, 'hours_model.pkl')
    print("Hours model saved to hours_model.pkl")

    # --- Train Model for Target Cost ---
    print("\n--- Training model for Target Cost ---")

    # 3. Split data for the cost model
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        features, target_cost, test_size=0.2, random_state=42
    )

    # 4. Initialize and train the XGBoost Regressor for cost
    cost_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training cost model...")
    cost_model.fit(X_train_c, y_train_c)

    # 5. Evaluate the cost model
    print("Evaluating cost model...")
    predictions_c = cost_model.predict(X_test_c)
    mae_c = mean_absolute_error(y_test_c, predictions_c)
    print(f"Mean Absolute Error (Cost Model): {mae_c:.2f} (in 10k units)")

    # 6. Save the cost model
    joblib.dump(cost_model, 'cost_model.pkl')
    print("Cost model saved to cost_model.pkl")
    
    print("\nModel training process complete!")

if __name__ == '__main__':
    train()

