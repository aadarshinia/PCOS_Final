from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

# Load saved model files
model = joblib.load(BASE_DIR / "pcos_risk_model.pkl")
features = joblib.load(BASE_DIR / "model_features.pkl")
importance_df = joblib.load(BASE_DIR / "feature_importance.pkl")


def predict_pcos_risk(user_data):

    """
    Predict PCOS risk using trained Random Forest model.
    """


    # Convert input into dataframe
    input_df = pd.DataFrame([user_data])


    # Add missing model features
    # with safe default values
    
# Add all missing model features at once
    missing_features = [
        feature for feature in features
        if feature not in input_df.columns
    ]

    if missing_features:
        missing_df = pd.DataFrame(
            0,
            index=input_df.index,
            columns=missing_features
        )
        input_df = pd.concat([input_df, missing_df], axis=1)

    # Ensure correct feature order
    input_df = input_df[features]


    # Prediction probability

    probability = model.predict_proba(input_df)[0][1]

    risk_score = float(
        round(probability * 100, 1)
    )


    # Explanation factors

    explanations = []


    if user_data.get("BMI",0) >= 25:
        explanations.append(
            "Elevated BMI"
        )


    if user_data.get("Menstrual_Irregularity",0) == 1:
        explanations.append(
            "Irregular menstrual cycles"
        )


    if user_data.get(
        "Testosterone_Level(ng/dL)",0
    ) > 45:

        explanations.append(
            "Elevated testosterone"
        )


    if user_data.get(
        "Antral_Follicle_Count",0
    ) >= 12:

        explanations.append(
            "High follicle count"
        )


    if user_data.get("LH",0) > 10:
        explanations.append(
            "Elevated LH"
        )


    if user_data.get("AMH",0) > 4:
        explanations.append(
            "Elevated AMH"
        )


    if user_data.get("Acne",0) == 1:
        explanations.append(
            "Acne reported"
        )


    if user_data.get("Hair_Growth",0) == 1:
        explanations.append(
            "Excess hair growth"
        )


    if user_data.get("Hair_Loss",0) == 1:
        explanations.append(
            "Hair loss symptoms"
        )


    if user_data.get("Weight_Gain",0) == 1:
        explanations.append(
            "Weight gain"
        )


    if user_data.get("Skin_Darkening",0) == 1:
        explanations.append(
            "Skin darkening"
        )


    if user_data.get(
        "Waist_Hip_Ratio",0
    ) > 0.85:

        explanations.append(
            "Elevated waist-to-hip ratio"
        )


    if user_data.get("RBS",0) > 110:
        explanations.append(
            "Raised blood glucose"
        )


    if user_data.get("Vitamin_D",100) < 20:
        explanations.append(
            "Low vitamin D"
        )


    if user_data.get(
        "Cycle_Length",0
    ) > 35:

        explanations.append(
            "Long menstrual cycle length"
        )


    return {

        "risk_score": risk_score,

        "important_factors": explanations

    }