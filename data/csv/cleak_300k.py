import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("student_performance_prediction_dataset-2 2.csv")

# 1. DROP LEAKAGE & IDS
df = df.drop(columns=["student_id", "grade_category", "final_grade"])

# 2. FILL CATEGORICAL NULLS
df["device_type"] = df["device_type"].fillna("None")
df["extracurriculars"] = df["extracurriculars"].fillna("None")

# 3. ENCODE
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
df["internet_access"] = df["internet_access"].map({"Yes": 1, "No": 0})
df["school_type"] = df["school_type"].map({"Private": 1, "Public": 0})
df["pass_fail"] = df["pass_fail"].map({"Pass": 1, "Fail": 0})

income_map = {"Low": 0, "Medium": 1, "High": 2}; edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
df["family_income"] = df["family_income"].map(income_map)
df["parent_education"] = df["parent_education"].map(edu_map)

df = pd.get_dummies(df, columns=["device_type", "extracurriculars"])

# --- STEP 3.5: REMOVE CONSTANT COLUMNS ---
# If a column has only 1 unique value, scaling it will result in NaN.
df = df.loc[:, df.nunique() > 1]

# 4. SPLIT
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 5. SCALE
cols_to_scale = [
    'age', 'study_hours', 'attendance', 'sleep_hours', 'previous_grade', 
    'assignments_completed', 'practice_tests_taken', 'group_study_hours', 
    'notes_quality_score', 'time_management_score', 'motivation_level', 
    'mental_health_score', 'screen_time', 'social_media_hours'
]

scaler = StandardScaler()
train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

# --- STEP 5.5: FINAL SANITIZATION ---
# Replace any unexpected Infinity with NaN, then fill all NaNs with 0
train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)

# 6. SAVE
train_df.to_csv("300k_train.csv", index=False)
test_df.to_csv("300k_test.csv", index=False)

print("Data is now 100% clean of NaNs and Infinity.")