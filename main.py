
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd


# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("Mental_Health_Model.pkl")


# =====================================================
# INPUT DATA MODEL
# =====================================================

class Student(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: str
    country: str
    academic_level: str
    most_used_platform: str
    purpose_of_use: str

    avg_daily_usage_hours: float = Field(
        ..., ge=0, le=24
    )

    daily_unlocks: int = Field(
        ..., ge=0
    )

    study_hours: float = Field(
        ..., ge=0, le=24
    )

    physical_activity_hours: float = Field(
        ..., ge=0, le=24
    )

    sleep_hours_per_night: float = Field(
        ..., ge=0, le=24
    )

    stress_level: str


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Mental Health Prediction API",
    description="Student Mental Health Score Prediction",
    version="1.0"
)


# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# RESPONSE MODEL
# =====================================================

class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():
    return {
        "message": "Welcome to the Mental Health Prediction API"
    }


# =====================================================
# COUNTRY GROUPING
# =====================================================

top_country = [
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France"
]


# =====================================================
# PREDICT
# =====================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: Student):

    # ---------------------------------------------
    # Group country
    # ---------------------------------------------

    country_group = (
        data.country
        if data.country in top_country
        else "others"
    )


    # ---------------------------------------------
    # Create DataFrame
    # ---------------------------------------------
    # IMPORTANT:
    # Mental_Health_Score is NOT included here.
    # It is the value we want the model to predict.
    #
    # Daily_Unlocks is also NOT included because
    # it was not one of the trained model features.
    # ---------------------------------------------

    input_row = pd.DataFrame([{

        "Study_Hours":
            data.study_hours,

        "Age":
            data.age,

        "Physical_Activity_Hours":
            data.physical_activity_hours,

        "Sleep_Hours_Per_Night":
            data.sleep_hours_per_night,

        "Avg_Daily_Usage_Hours":
            data.avg_daily_usage_hours,

        "Stress_Level":
            data.stress_level,

        "Most_Used_Platform":
            data.most_used_platform,

        "grouped_Country":
            country_group,

        "Gender":
            data.gender,

        "Academic_Level":
            data.academic_level,

        "Purpose_Of_Use":
            data.purpose_of_use
    }])


    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    prediction = model.predict(input_row)[0]


    # ---------------------------------------------
    # Return result
    # ---------------------------------------------

    return PredictionResponse(
        predicted_mental_health_score=round(
            float(prediction),
            2
        )
    )
    