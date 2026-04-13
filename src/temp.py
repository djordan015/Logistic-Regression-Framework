import pandas as pd
import numpy as np

def generate_perfect_data(filename="scaled_data.csv", samples=1000, features=5):
    # 1. Generate features from a Standard Normal Distribution (mean=0, std=1)
    X = np.random.randn(samples, features)
    
    # 2. Generate a Target based on a simple linear combination
    # If the sum of features is > 0, label is 1, else 0.
    # This creates a perfectly separable dataset for Logistic Regression.
    y = (np.sum(X, axis=1) > 0).astype(int)
    
    # 3. Create DataFrame
    cols = [f"X{i+1}" for i in range(features)]
    df = pd.DataFrame(X, columns=cols)
    df['Target'] = y
    
    # 4. Save to CSV
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {samples} samples.")
    print(f"Feature Means: {np.mean(X, axis=0)}")
    print(f"Feature Std: {np.std(X, axis=0)}")

generate_perfect_data()

