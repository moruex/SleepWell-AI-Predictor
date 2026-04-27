from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
from pydantic import BaseModel
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

cognitive_model = joblib.load('models/cognitive_model.joblib')
cognitive_columns = joblib.load('models/cognitive_columns.joblib')
cognitive_scaler = joblib.load('models/cognitive_scaler.joblib')

disorder_model = joblib.load('models/disorder_model.joblib')
disorder_columns = joblib.load('models/disorder_model_columns.joblib')
disorder_scaler = joblib.load('models/disorder_scaler.joblib')
disorder_encoding = joblib.load('models/disorder_encoding.joblib')

class SleepFeatures(BaseModel):
    age: int
    occupation: str
    weight_kg: float
    height_cm: float
    sleep_duration_hrs: float
    rem_percentage: float
    deep_sleep_percentage: float
    sleep_latency_mins: float
    wake_episodes_per_night: int
    caffeine_mg_before_bed: int
    alcohol_units_before_bed: float
    screen_time_before_bed_mins: int
    exercise_day: int
    steps_that_day: int
    nap_duration_mins: int
    stress_score: float
    work_hours_that_day: float
    mental_health_condition: str
    heart_rate_resting_bpm: int
    sleep_aid_used: str
    shift_work: str
    day_type: str

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(features: SleepFeatures):
    data = features.model_dump()
    df = pd.DataFrame([data])

    df['bmi'] = features.weight_kg / ((features.height_cm / 100) ** 2)

    df['mental_health_condition'] = disorder_encoding['mental_mapping'].get(features.mental_health_condition, 0)
    df['day_type'] = disorder_encoding['day_type_mapping'].get(features.day_type, 0)
    df['sleep_aid_used'] = 1 if features.sleep_aid_used.lower() == "yes" else 0
    df['shift_work'] = 1 if features.shift_work.lower() == "yes" else 0

    df_encoded = pd.get_dummies(df, columns=['occupation'], prefix='occ')

    df_cog = df_encoded.reindex(columns=cognitive_columns, fill_value=0)
    df_cog_scaled = cognitive_scaler.transform(df_cog)
    cog_pred = cognitive_model.predict(df_cog_scaled)[0]

    df_dis = df_encoded.reindex(columns=disorder_columns, fill_value=0)
    df_dis_scaled = disorder_scaler.transform(df_dis)
    dis_pred = disorder_model.predict(df_dis_scaled)[0]

    return {
        "cognitive_performance_score": round(float(cog_pred), 2),
        "sleep_disorder_risk": int(dis_pred),
        "calculated_bmi": round(float(df['bmi'][0]), 2)
    }