# app2.py
import os
import threading
import webbrowser
from pathlib import Path
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from pwdlib import PasswordHash
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from groq import Groq
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from fastapi import Depends
from database import SessionLocal, User, SymptomEntryDB

# Import prediction function from predict.py
from predict import predict_pcos_risk

# Load environment variables (.env file)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="PCOS-AI Backend API")
# Password hashing
password_hash = PasswordHash.recommended()

# Session secret: set SESSION_SECRET in your .env file.
# The fallback is for local development only.
SESSION_SECRET = os.getenv(
    "SESSION_SECRET",
    "local-development-secret-change-this"
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=False,  # Local development only; use HTTPS in production.
)
def get_db():
    """Provide a database session for one request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
def hash_password(password: str) -> str:
    """Hash a password before storing it in the database."""
    return password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Check a submitted password against its stored hash."""
    return password_hash.verify(password, hashed_password)


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
):
    """Return the logged-in user or reject the request."""
    user_id = request.session.get("user_id")

    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Please log in to continue."
        )

    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        request.session.clear()
        raise HTTPException(
            status_code=401,
            detail="Account not found. Please log in again."
        )

    return user

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
    period_start: str = ""
    period_end: str = ""
    pain: int = 0
    acne: str = "none"
    mood: str = "neutral"
    energy: int = 0
    weight: Optional[float] = None
    sleep: float = 0
    medication: str = ""
    notes: str = ""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str

# =========================================================
# 🚀 ROUTING & PAGE ENDPOINTS
# =========================================================
@app.get("/login")
def serve_login_page():
    """Serve the login and signup page."""
    return FileResponse(str(BASE_DIR / "login.html"))


@app.post("/api/auth/signup")
def signup(
    data: SignupRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Create an account and sign the new user in."""
    name = data.name.strip()
    email = data.email.strip().lower()
    password = data.password

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Please enter your name."
        )

    if len(password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters."
        )

    existing_user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    user = User(
        name=name,
        email=email,
        hashed_password=hash_password(password),
    )

    try:
        db.add(user)
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    request.session.clear()
    request.session["user_id"] = user.id

    return {
        "message": "Account created successfully.",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        },
    }


@app.post("/api/auth/login")
def login(
    data: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Verify credentials and start a login session."""
    email = data.email.strip().lower()

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if user is None or not verify_password(
        data.password,
        user.hashed_password,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    request.session.clear()
    request.session["user_id"] = user.id

    return {
        "message": "Logged in successfully.",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        },
    }


@app.post("/api/auth/logout")
def logout(request: Request):
    """End the current login session."""
    request.session.clear()
    return {"message": "Logged out successfully."}


@app.get("/api/auth/me")
def get_my_account(
    current_user: User = Depends(get_current_user),
):
    """Return the account associated with the current session."""
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
    }

@app.get("/")
def serve_landing_page():
    """Serves the main landing page directly upon startup."""
    return FileResponse(str(BASE_DIR / "landing.html"))

@app.get("/app")
def serve_app_page(
    request: Request,
    db: Session = Depends(get_db),
):
    """Serve the dashboard only to logged-in users."""
    user_id = request.session.get("user_id")

    if user_id is None:
        return RedirectResponse(
            url="/login",
            status_code=303,
        )

    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        request.session.clear()
        return RedirectResponse(
            url="/login",
            status_code=303,
        )

    return FileResponse(str(BASE_DIR / "index.html"))

@app.get("/health")
def health_check():
    """Simple health endpoint for deployment platforms."""
    return {"status": "ok"}

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


def symptom_to_dict(entry: SymptomEntryDB) -> dict:
    """Convert a database symptom entry into dashboard-friendly data."""
    return {
        "id": entry.id,
        "date": entry.date,
        "period_start": entry.period_start,
        "period_end": entry.period_end,
        "pain": entry.pain,
        "acne": entry.acne,
        "mood": entry.mood,
        "energy": entry.energy,
        "weight": entry.weight,
        "sleep": entry.sleep,
        "medication": entry.medication,
        "notes": entry.notes,
        "period_duration": entry.period_duration,
        "suggestions": entry.suggestions or [],
        "created_at": (
            entry.created_at.isoformat()
            if entry.created_at else None
        ),
    }


def generate_symptom_suggestions(entry: SymptomEntry) -> list:
    """Generate the existing educational suggestions for a symptom entry."""
    suggestions = []

    if entry.pain >= 7:
        suggestions.append(
            "🩺 High pain level reported. If severe or persistent, "
            "consult a medical professional."
        )
    elif entry.pain >= 4:
        suggestions.append(
            "🩺 Moderate pelvic/abdominal pain noted. "
            "Consider discussing persistent symptoms with a healthcare professional."
        )

    if entry.sleep < 7:
        suggestions.append(
            "😴 Less than 7 hours of sleep logged. "
            "Tracking sleep may help you notice patterns."
        )

    if entry.energy <= 4:
        suggestions.append(
            "⚡ Low energy reported today. Consider noting sleep, "
            "hydration, and meals to discuss any ongoing concerns with a professional."
        )

    if entry.mood in ["Stressed", "Anxious", "Low"]:
        suggestions.append(
            "🧠 Emotional changes noticed. Relaxation strategies "
            "or talking with someone you trust may be helpful."
        )

    if entry.acne in ["Moderate", "Severe"]:
        suggestions.append(
            "🧴 You logged acne symptoms. Tracking changes over time "
            "may help you discuss them with a healthcare professional."
        )

    return suggestions


@app.get("/api/tracker")
def get_symptoms(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return only symptom entries belonging to the logged-in user."""
    entries = (
        db.query(SymptomEntryDB)
        .filter(SymptomEntryDB.user_id == current_user.id)
        .order_by(SymptomEntryDB.id.asc())
        .all()
    )

    return [symptom_to_dict(entry) for entry in entries]


@app.post("/api/tracker")
def add_symptom(
    entry: SymptomEntry,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a symptom entry linked to the logged-in user."""
    try:
        d_start = datetime.strptime(
            entry.period_start, "%Y-%m-%d"
        )
        d_end = datetime.strptime(
            entry.period_end, "%Y-%m-%d"
        )

        period_duration = max(
            1, (d_end - d_start).days + 1
        )
    except (ValueError, TypeError):
        period_duration = 1

    suggestions = generate_symptom_suggestions(entry)

    new_entry = SymptomEntryDB(
        user_id=current_user.id,
        date=datetime.now().strftime("%Y-%m-%d"),
        period_start=entry.period_start,
        period_end=entry.period_end,
        pain=entry.pain,
        acne=entry.acne,
        mood=entry.mood,
        energy=entry.energy,
        weight=entry.weight,
        sleep=entry.sleep,
        medication=entry.medication,
        notes=entry.notes,
        period_duration=period_duration,
        suggestions=suggestions,
    )

    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)

    return {
        "status": "success",
        "entry": symptom_to_dict(new_entry),
    }


@app.delete("/api/tracker/{entry_id}")
def delete_symptom(
    entry_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete an entry only if it belongs to the logged-in user."""
    entry = (
        db.query(SymptomEntryDB)
        .filter(
            SymptomEntryDB.id == entry_id,
            SymptomEntryDB.user_id == current_user.id,
        )
        .first()
    )

    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="Symptom entry not found."
        )

    deleted_entry = symptom_to_dict(entry)

    db.delete(entry)
    db.commit()

    return {
        "status": "success",
        "deleted": deleted_entry,
    }


@app.delete("/api/tracker")
def clear_symptoms(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Clear only the logged-in user's symptom history."""
    (
        db.query(SymptomEntryDB)
        .filter(SymptomEntryDB.user_id == current_user.id)
        .delete(synchronize_session=False)
    )

    db.commit()

    return {
        "status": "success",
        "message": "Your symptom history was cleared.",
    }


@app.post("/api/chat")
def chat_assistant(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    AI Educational Health Assistant powered by Groq (llama-3.3-70b-versatile).
    Injects recent user symptom history directly into the system prompt for contextualized answers.
    """
    if not client:
        raise HTTPException(status_code=500, detail="Groq API key missing. Please check your .env file.")

    recent_entries = (
    db.query(SymptomEntryDB)
    .filter(SymptomEntryDB.user_id == current_user.id)
    .order_by(SymptomEntryDB.id.desc())
    .limit(5)
    .all()
    )
    if recent_entries:
        recent_entries.reverse()

        recent_data = [
            symptom_to_dict(entry)
            for entry in recent_entries
        ]

        df = pd.DataFrame(recent_data)
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
    """Opens the landing page in a local browser when running locally."""
    if os.getenv("RENDER") or os.getenv("PORT"):
        return
    webbrowser.open_new("http://127.0.0.1:8000/")

if __name__ == "__main__":
    import uvicorn

    # Schedule browser launch after server initialization for local runs
    threading.Timer(1.2, open_browser).start()

    # Run Uvicorn ASGI server with Render-friendly host/port settings
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)