import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load the dataset
print("Loading data...")
try:
    df = pd.read_csv("first inten project.csv")
except FileNotFoundError:
    print("Error: 'first inten project.csv' not found. Please ensure the file is in the same directory.")
    exit()

# Data Cleaning and Feature Engineering
df.columns = df.columns.str.strip()
df.rename(columns={'average price ': 'average price'}, inplace=True)

# Create the combined features you requested
df['total_nights'] = df['number of weekend nights'] + df['number of week nights']
df['total_guests'] = df['number of adults'] + df['number of children']

# Drop the original columns used for engineering
df.drop(columns=['number of weekend nights', 'number of week nights',
                 'number of adults', 'number of children'], inplace=True)

# Drop rows with any missing values that might exist in other columns
df.dropna(inplace=True)

# Define the features for the model
features_to_use = [
    'average price', 
    'total_guests',
    'total_nights',
    'room type'
]

target = 'booking status'

X = df[features_to_use]
y = df[target]

# Identify numerical and categorical features
numerical_features = ['average price', 'total_guests', 'total_nights']
categorical_features = ['room type']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the full model pipeline
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train and evaluate the model
print("Training and evaluating the Logistic Regression model...")
log_reg_pipeline.fit(X_train, y_train)
y_pred = log_reg_pipeline.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(log_reg_pipeline, 'models/log_reg_pipeline.pkl')
print("\nModel saved successfully as 'models/log_reg_pipeline.pkl'.")