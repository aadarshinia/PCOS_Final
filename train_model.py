import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)


# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv(
    "pcos_dataset_combined.csv"
)


print("Original dataset shape:")
print(df.shape)


print("\nOriginal columns:")
print(df.columns.tolist())



# =====================================================
# 2. CLEAN TARGET COLUMN
# =====================================================

# Handle Kaggle dataset naming if still present

if "PCOS(Y/N)" in df.columns:

    df["PCOS_Diagnosis"] = (
        df["PCOS(Y/N)"]
        .map({
            "Y": 1,
            "N": 0
        })
    )


# Convert target to numeric

df["PCOS_Diagnosis"] = pd.to_numeric(
    df["PCOS_Diagnosis"],
    errors="coerce"
)


# Remove rows without diagnosis

df = df.dropna(
    subset=["PCOS_Diagnosis"]
)


df["PCOS_Diagnosis"] = df["PCOS_Diagnosis"].astype(int)



# =====================================================
# 3. REMOVE IDENTIFIER COLUMNS
# =====================================================

remove_columns = [

    "SI.No",

    "Patient File No",

    "Blood Group"

]


df.drop(
    columns=remove_columns,
    inplace=True,
    errors="ignore"
)



# =====================================================
# 4. CONVERT CATEGORICAL DATA
# =====================================================

df = pd.get_dummies(
    df,
    drop_first=True
)



# =====================================================
# 5. HANDLE MISSING VALUES
# =====================================================

df = df.fillna(
    df.median(numeric_only=True)
)



print("\nCleaned dataset shape:")
print(df.shape)



# =====================================================
# 6. SPLIT FEATURES AND TARGET
# =====================================================

X = df.drop(
    "PCOS_Diagnosis",
    axis=1
)


y = df[
    "PCOS_Diagnosis"
]


print("\nModel Features:")
print(list(X.columns))



# =====================================================
# 7. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# =====================================================
# 8. RANDOM FOREST MODEL
# =====================================================

model = RandomForestClassifier(

    n_estimators=500,

    class_weight="balanced",

    random_state=42

)



model.fit(

    X_train,

    y_train

)



# =====================================================
# 9. TEST MODEL
# =====================================================

predictions = model.predict(
    X_test
)


probabilities = model.predict_proba(
    X_test
)[:,1]



accuracy = accuracy_score(

    y_test,

    predictions

)


roc_auc = roc_auc_score(

    y_test,

    probabilities

)



print("\n============================")
print("MODEL PERFORMANCE")
print("============================")


print(
    "Accuracy:",
    accuracy
)


print(
    "ROC-AUC:",
    roc_auc
)


print(
    "\nClassification Report:"
)


print(
    classification_report(
        y_test,
        predictions
    )
)


print(
    "\nConfusion Matrix:"
)


print(
    confusion_matrix(
        y_test,
        predictions
    )
)



# =====================================================
# 10. FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})


importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)



print(
    "\nFeature Importance:"
)


print(
    importance_df.to_string(index=False)
)



# =====================================================
# 11. SAVE MODEL FILES
# =====================================================

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



print(
    "\n============================"
)

print(
    "New model saved successfully!"
)

print(
    "Features saved:",
    len(X.columns)
)

print(
    "============================"
)