import pandas as pd
import numpy as np

def generate_model_data(train_size=1000, test_size=200, features=5, filename_prefix="cancer_sim"):
    """
    Generates two distinct CSV files (train and test) with scaled, 
    linearly separable data to test Logistic Regression.
    """
    total_samples = train_size + test_size
    
    # 1. Generate features from a Standard Normal Distribution (mean=0, std=1)
    X = np.random.randn(total_samples, features)
    
    # 2. Create a hidden "rule" for the labels
    # We use a weight vector to decide the target: y = sign(sum(W * X) + bias)
    # This ensures the data is mathematically "learnable"
    true_weights = np.random.uniform(-1, 1, features)
    true_bias = 0.5
    
    # Calculate linear combination and apply a sigmoid-like logic for labels
    logits = np.dot(X, true_weights) + true_bias
    y = (logits > 0).astype(int)
    
    # 3. Create DataFrame
    cols = [f"X{i+1}" for i in range(features)]
    df = pd.DataFrame(X, columns=cols)
    df['Target'] = y
    
    # 4. Split into Train and Unseen Test sets
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    # 5. Save to CSVs
    train_name = f"{filename_prefix}_train.csv"
    test_name = f"{filename_prefix}_test.csv"
    
    df_train.to_csv(train_name, index=False)
    df_test.to_csv(test_name, index=False)
    
    print(f"Successfully generated datasets:")
    print(f"  - Training: {train_name} ({train_size} samples)")
    print(f"  - Unseen Test: {test_name} ({test_size} samples)")
    print(f"  - Feature count: {features}")
    
    return train_name, test_name

# Example usage:
generate_model_data(train_size=1500, test_size=500, features=10)