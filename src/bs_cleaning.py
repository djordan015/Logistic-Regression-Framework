import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('bs.csv')

# Remove the 'id' column
df_processed = df.drop(columns=['id'])

# Split the data into train (80%) and test (20%) sets
# 'stratify' ensures the distribution of classes is preserved
train_df, test_df = train_test_split(
    df_processed, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_processed['diagnosis']
)

# Save the resulting sets to CSV files
train_df.to_csv('train_set.csv', index=False)
test_df.to_csv('test_set.csv', index=False)

print(f"Training set size: {train_df.shape}")
print(f"Testing set size: {test_df.shape}")