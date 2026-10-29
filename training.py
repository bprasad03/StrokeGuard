import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE

# --- 1. MOCK DATA LOADING (Replace with your actual data loading) ---
# Since the original CSV is not available, we create a mock DataFrame
# that has the same columns and data types as your original data
import os

# Get path to dataset in the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "healthcare-dataset-stroke-data.csv")
data = pd.read_csv(csv_path)
# We load the full dataset for demonstration. In a real scenario, you'd load the full, preprocessed dataset here.
# For simplicity, we are using this small mock dataset to fit the pipeline structure.
# In a real scenario, this mock data should be replaced by:
# df = pd.read_csv("your_original_full_dataset.csv")
df = pd.DataFrame(data)
df = df.drop('id',axis =1)
# To mimic a real-world scenario, we'll quickly create a larger, imbalanced dataset.
# In your actual project, you would load your full "healthcare-dataset-stroke-data.csv"
# and perform the split as in your notebook (but with the correct CSV path).

# Define feature and target variables
X = df.drop('stroke', axis=1)
y = df['stroke']

# --- 2. RECREATE PIPELINE COMPONENTS ---
cat_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type', 'smoking_status']
num_cols = ['age', 'avg_glucose_level', 'bmi']

# Numerical Transformer (Imputer + Normalizer)
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('power', PowerTransformer(method='yeo-johnson', standardize=True))
])

# Categorical Transformer (Imputer + One-Hot Encoder)
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column Transformer (Preprocess all features)
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
], remainder='passthrough')


# --- 3. CREATE FINAL PIPELINE (Preprocessor + SMOTE + LDA Model) ---
# NOTE: The LDA model is the one chosen in your analysis.
# The `SMOTE` step will be applied *after* preprocessing in the final pipeline.
# Since SMOTE changes the number of rows, we define a full pipeline including the model.

final_lda_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', LinearDiscriminantAnalysis())
])

# --- 4. TRAIN AND SAVE THE MODEL ---
print("Fitting the final LDA pipeline...")
# Fit the pipeline to the complete dataset (or training set in complex cases)
# We use the small mock data here for demonstration purposes.
final_lda_pipeline.fit(X, y)
print("Pipeline fitted successfully.")

# Save the entire fitted pipeline object (preprocessor + SMOTE + LDA)
with open('pipeline.pkl', 'wb') as file:
    pickle.dump(final_lda_pipeline, file)

print("Pipeline saved to pipeline.pkl. Ready for deployment.")

# Note for deployment: The 'gender' column in the notebook's initial data dump
# contained 'Other'. We must ensure the training data used to fit the
# OneHotEncoder includes all possible categories if not ignored.
# The mock data above includes 'Other' to ensure OHE works correctly.