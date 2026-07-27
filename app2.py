# app2.py
import os
import threading
import webbrowser
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from groq import Groq
from dotenv import load_dotenv

# Import prediction function from predict.py
from predict import predict_pcos_risk

# Load environment variables (.env file)
load_dotenv()

app = FastAPI(title="PCOS-AI Backend API")

# Enable CORS for local HTML/JS frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# In-memory storage for symptom tracking (session store)
symptom_store = []


# =========================================================
# 📋 DATA SCHEMAS (Pydantic Models)
# =========================================================

class ProfileData(BaseModel):

    # -------------------------
    # Basic Information
    # -------------------------

    age: int

    height: float

    weight: float

    # -------------------------
    # Hormonal Tests
    # -------------------------

    testosterone: float = 35

    lh: float = 5

    fsh: float = 5

    amh: float = 4

    tsh: float = 2

    rbs: float = 90

    # -------------------------
    # Ultrasound
    # -------------------------

    follicle_count: int = 8

    # -------------------------
    # Menstrual Health
    # -------------------------

    cycle_length: int = 28

    irregular_periods: bool = False

    # -------------------------
    # Symptoms
    # -------------------------

    acne: bool = False

    hair_growth: bool = False

    hair_loss: bool = False

    weight_gain: bool = False

    skin_darkening: bool = False

    # -------------------------
    # Lifestyle
    # -------------------------

    fast_food: bool = False

    exercise: bool = True

    vitamin_d: float = 30

    waist_hip_ratio: float = 0.85

class RiskAssessmentRequest(BaseModel):
    profile: ProfileData

class SymptomEntry(BaseModel):
    period_start: str
    period_end: str
    pain: int
    acne: str
    mood: str
    energy: int
    weight: float
    sleep: float
    medication: str = ""
    notes: str = ""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]


# =========================================================
# 🚀 ROUTING & PAGE ENDPOINTS
# =========================================================

@app.get("/")
def serve_landing_page():
    """Serves the main landing page directly upon startup."""
    return FileResponse("landing.html")

@app.get("/app")
def serve_app_page():
    """Serves the main application dashboard (index.html)."""
    return FileResponse("index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Silences 404 logs for browser favicon requests."""
    from fastapi import Response
    return Response(status_code=204)


# =========================================================
# ⚡ API BACKEND ENDPOINTS
# =========================================================

@app.post("/api/predict")
def calculate_pcos_risk(req: RiskAssessmentRequest):
    """
    Computes BMI and passes clinical inputs to the trained Random Forest model.
    """
    height_m = req.profile.height / 100.0
    bmi = round(req.profile.weight / (height_m ** 2), 1)

    user_data = {

    "Age": req.profile.age,

    "BMI": bmi,

    "Menstrual_Irregularity": int(req.profile.irregular_periods),

    "Testosterone_Level(ng/dL)": req.profile.testosterone,

    "Antral_Follicle_Count": req.profile.follicle_count,

    "LH": req.profile.lh,

    "FSH": req.profile.fsh,

    "Cycle_Length": req.profile.cycle_length,

    "Acne": int(req.profile.acne),

    "Hair_Growth": int(req.profile.hair_growth),

    "Hair_Loss": int(req.profile.hair_loss),

    "AMH": req.profile.amh,

    "Weight_Gain": int(req.profile.weight_gain),

    "Skin_Darkening": int(req.profile.skin_darkening),

    "Waist_Hip_Ratio": req.profile.waist_hip_ratio,

    "TSH": req.profile.tsh,

    "RBS": req.profile.rbs,

    "Fast_Food": int(req.profile.fast_food),

    "Exercise": int(req.profile.exercise),

    "Vitamin_D": req.profile.vitamin_d

}
    try:
        prediction_res = predict_pcos_risk(user_data)
        
        return {
            "risk_score": prediction_res["risk_score"],
            "bmi": bmi,
            "important_factors": prediction_res["important_factors"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tracker")
def get_symptoms():
    """Retrieve all tracked symptom entries."""
    return symptom_store


@app.post("/api/tracker")
def add_symptom(entry: SymptomEntry):
    """Log a daily symptom entry and automatically generate health insights."""
    entry_dict = entry.dict()
    entry_dict["date"] = datetime.now().strftime("%Y-%m-%d")

    try:
        d_start = datetime.strptime(entry.period_start, "%Y-%m-%d")
        d_end = datetime.strptime(entry.period_end, "%Y-%m-%d")
        period_duration = (d_end - d_start).days + 1
    except Exception:
        period_duration = 1
    entry_dict["period_duration"] = period_duration

    suggestions = []
    if entry.pain >= 7:
        suggestions.append("🩺 High pain level reported. If severe or persistent, consult a medical professional.")
    elif entry.pain >= 4:
        suggestions.append("🩺 Moderate pelvic/abdominal pain noted. Gentle stretching or heat therapy may help ease discomfort.")
        
    if entry.sleep < 7:
        suggestions.append("😴 Less than 7 hours of sleep logged. Sleep regulation helps maintain balanced cortisol and insulin levels.")
        
    if entry.energy <= 4:
        suggestions.append("⚡ Low energy reported today. Ensure adequate hydration and balanced meals with complex carbohydrates.")
        
    if entry.mood in ["Stressed", "Anxious", "Low"]:
        suggestions.append("🧠 Emotional changes noticed. Mindfulness or breathing techniques can support stress management.")
        
    if entry.acne in ["Moderate", "Severe"]:
        suggestions.append("🧴 Persistent skin flares can be linked with androgen level variations. Keep tracking trends.")

    entry_dict["suggestions"] = suggestions
    symptom_store.append(entry_dict)
    
    return {"status": "success", "entry": entry_dict}


@app.delete("/api/tracker/{index}")
def delete_symptom(index: int):
    """Delete a specific symptom entry by list index."""
    if 0 <= index < len(symptom_store):
        deleted = symptom_store.pop(index)
        return {"status": "success", "deleted": deleted}
    raise HTTPException(status_code=404, detail="Entry index not found")


@app.delete("/api/tracker")
def clear_symptoms():
    """Clear all symptom entries."""
    global symptom_store
    symptom_store = []
    return {"status": "success", "message": "All symptom history cleared"}


@app.post("/api/chat")
def chat_assistant(request: ChatRequest):
    """
    AI Educational Health Assistant powered by Groq (llama-3.3-70b-versatile).
    Injects recent user symptom history directly into the system prompt for contextualized answers.
    """
    if not client:
        raise HTTPException(status_code=500, detail="Groq API key missing. Please check your .env file.")

    if len(symptom_store) > 0:
        df = pd.DataFrame(symptom_store).tail(5)
        user_context = df.to_string(index=False)
    else:
        user_context = "No symptom tracking history recorded yet."

    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful PCOS educational AI assistant.\n"
            "Your knowledge base aligns with clinical guidelines from: International Evidence-based Guideline for PCOS, NHS, NICE, ACOG, WHO, and the Endocrine Society.\n\n"
            f"The user's recent symptom tracker history is:\n{user_context}\n\n"
            "Strict Guidelines:\n"
            "1. Provide clear, evidence-based educational explanations.\n"
            "2. Never issue a medical diagnosis.\n"
            "3. Never declare that the user definitely has or does not have PCOS.\n"
            "4. Acknowledge medical limitations when data or evidence is unclear.\n"
            "5. Encourage medical consultation with a healthcare provider for diagnosis or treatment decisions.\n"
            "6. Reference the user's symptom history only to tailor educational insights."
        )
    }

    full_messages = [system_prompt] + [m.dict() for m in request.messages]

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=full_messages,
            temperature=0.3,
            max_tokens=700,
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# 🏁 SCRIPT EXECUTION & AUTO-BROWSER LAUNCH
# =========================================================

def open_browser():
    """Opens the landing page in default web browser once server is live."""
    webbrowser.open_new("http://127.0.0.1:8000/")

if __name__ == "__main__":
    import uvicorn
    
    # Schedule browser launch after server initialization
    threading.Timer(1.2, open_browser).start()
    
    # Run Uvicorn ASGI server
    uvicorn.run("app2:app", host="127.0.0.1", port=8000, reload=True)