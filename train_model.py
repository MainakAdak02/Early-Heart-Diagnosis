import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- 1. LOAD DATA ---
file_name = 'heart_disease_uci.csv'
print(f"Loading {file_name}...")
df = pd.read_csv(file_name)

# --- 2. STRICT CLEANING (Force 13 Columns) ---
# We manually map text to numbers to ensure the model matches the website EXACTLY.

# Drop ID/Dataset if they exist
df = df.drop(['id', 'dataset'], axis=1, errors='ignore')

# MAP: Sex
df['sex'] = df['sex'].astype(str).str.lower().map({'male': 1, 'female': 0, 'm': 1, 'f': 0})

# MAP: Chest Pain (cp)
# Adjust these strings to match your CSV file exactly if needed
cp_map = {'typical angina': 1, 'atypical angina': 2, 'non-anginal pain': 3, 'asymptomatic': 4}
df['cp'] = df['cp'].map(cp_map).fillna(4) # Default to 4 if unknown

# MAP: FBS (True/False -> 1/0)
df['fbs'] = df['fbs'].astype(str).str.lower().map({'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0)

# MAP: RestECG
restecg_map = {'normal': 0, 'st-t wave abnormality': 1, 'left ventricular hypertrophy': 2}
df['restecg'] = df['restecg'].map(restecg_map).fillna(0)

# MAP: Exang (Exercise Angina)
df['exang'] = df['exang'].astype(str).str.lower().map({'yes': 1, 'no': 0, 'true': 1, 'false': 0}).fillna(0)

# MAP: Slope
slope_map = {'upsloping': 1, 'flat': 2, 'downsloping': 3}
df['slope'] = df['slope'].map(slope_map).fillna(2)

# MAP: Thal
thal_map = {'normal': 3, 'fixed defect': 6, 'reversable defect': 7}
df['thal'] = df['thal'].map(thal_map).fillna(3)

# Handle Numeric Columns (Force to numbers, coerce errors to NaN)
for col in ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numerical values with average
df = df.fillna(df.mean())

# --- 3. PREPARE TARGET ---
# Ensure target is 0 or 1
if 'num' in df.columns:
    y = (df['num'] > 0).astype(int)
    X = df.drop('num', axis=1)
elif 'target' in df.columns:
    y = df['target']
    X = df.drop('target', axis=1)
else:
    print("Error: Target column not found")
    exit()

# Verify we have exactly 13 columns
expected_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = X[expected_cols] # Force order

print(f"Training with {X.shape[1]} columns: {list(X.columns)}")

# --- 4. TRAIN & SAVE ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"New Model Accuracy: {model.score(X_test, y_test)*100:.2f}%")

pickle.dump(model, open('heart_model.pkl', 'wb'))
print("SUCCESS: New heart_model.pkl saved!")