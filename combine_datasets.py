import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("pcos_dataset_combined.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ==========================================================
# Handle Missing Values
# ==========================================================

# Fill numeric missing values using the median
df = df.fillna(df.median(numeric_only=True))

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ==========================================================
# Split Features and Target
# ==========================================================

X = df.drop("PCOS_Diagnosis", axis=1)
y = df["PCOS_Diagnosis"]

print("\nTraining Features:")
print(X.columns.tolist())

# ==========================================================
# Train/Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples : {len(X_test)}")

# ==========================================================
# Train Random Forest
# ==========================================================

model = RandomForestClassifier(
    n_estimators=800,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================================
# Predictions
# ==========================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

# ==========================================================
# Evaluation
# ==========================================================

accuracy = accuracy_score(y_test, predictions)

roc = roc_auc_score(y_test, probabilities)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {roc:.4f}")

print("\nClassification Report")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

# ==========================================================
# Feature Importance
# ==========================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(importance_df.to_string(index=False))

# ==========================================================
# Save Model Files
# ==========================================================

joblib.dump(
    model,
    "pcos_risk_model.pkl"
)

joblib.dump(
    list(X.columns),
    "model_features.pkl"
)

joblib.dump(
    importance_df,
    "feature_importance.pkl"
)

print("\n==============================")
print("Model exported successfully!")
print("==============================")