import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC
import joblib

# ------------------ Load Data ------------------
df = pd.read_csv("first inten project.csv")

date_col = 'date of reservation'
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df['year'] = df[date_col].dt.year
df['month'] = df[date_col].dt.month
df.drop(columns=[date_col], inplace=True)

# Create combined features
df['total_nights'] = df['number of weekend nights'] + df['number of week nights']
df['total_guests'] = df['number of adults'] + df['number of children']

# Drop unnecessary columns
df.drop(columns=['number of weekend nights', 'number of week nights',
                 'number of adults', 'number of children',
                 'weekday'], inplace=True, errors='ignore')

# Drop missing rows
df.dropna(inplace=True)
df.columns = df.columns.str.strip()
# Target mapping
df['target'] = df['booking status'].map({'Canceled': 1, 'Not_Canceled': 0})
target = 'target'

# ------------------ Select Features ------------------
selected_features = [
    'average price',
    'total_guests',
    'total_nights',
    'room type',
    'type of meal',
    'market segment type',
    'month',
    'year'
]

X = df[selected_features]
y = df[target]

# ------------------ Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------ Preprocessing ------------------
numeric_cols = ['average price', 'total_guests', 'total_nights', 'month', 'year']
categorical_cols = ['room type', 'type of meal', 'market segment type']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Fit + transform
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# Save preprocessor
joblib.dump(preprocessor, "preprocessor.pkl")

# ------------------ Handle imbalance with SMOTENC ------------------
cat_start = len(numeric_cols)
categorical_indices = list(range(cat_start, cat_start + len(categorical_cols)))

sm = SMOTENC(categorical_features=categorical_indices, random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train_trans, y_train)

# ------------------ Random Forest ------------------
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)

y_pred = rf.predict(X_test_trans)
y_proba = rf.predict_proba(X_test_trans)[:,1]

print("\n--- Random Forest Evaluation ---")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

joblib.dump(rf, "random_forest_model.pkl")

# ------------------ Logistic Regression ------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_bal, y_train_bal)

y_pred = log_reg.predict(X_test_trans)
y_proba = log_reg.predict_proba(X_test_trans)[:,1]

print("\n--- Logistic Regression Evaluation ---")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

joblib.dump(log_reg, "logistic_regression_model.pkl")

# ------------------ Save Selected Features ------------------
df_selected = df[selected_features + ['booking status']]
df_selected.to_csv("selected_features.csv", index=False)

print("\n✅ Models & selected features saved successfully.")
