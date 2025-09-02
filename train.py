# train.py
# ---------------------------------
# Prepare data, train model, save model + scaler + encoder
# ---------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Step 2: Encode target variable (crop names → numbers)
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Save mapping of crops for later use
crop_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nCrop Mapping:", crop_mapping)

# Step 3: Split features and target
X = df.drop(['label', 'label_encoded'], axis=1)
y = df['label_encoded']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Random Forest Model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

# Step 7: Evaluate
y_pred = rf.predict(X_test_scaled)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Step 8: Save model, scaler, and encoder
joblib.dump(rf, "crop_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n🎉 Model, scaler, and encoder saved successfully!")
