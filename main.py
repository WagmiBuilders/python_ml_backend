from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import datetime
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import shutil
from typing import List, Dict, Optional
import os
from pydantic import BaseModel, Field
from fastapi import Body
from datetime import date
from enum import Enum


class WindDirection(str, Enum):
    N = "N"
    NNE = "NNE"
    NE = "NE"
    ENE = "ENE"
    E = "E"
    ESE = "ESE"
    SE = "SE"
    SSE = "SSE"
    S = "S"
    SSW = "SSW"
    SW = "SW"
    WSW = "WSW"
    W = "W"
    WNW = "WNW"
    NW = "NW"
    NNW = "NNW"


class YesNo(str, Enum):
    Yes = "Yes"
    No = "No"


app = FastAPI()

file_name = os.getenv("DATA_PATH", "data/data.csv")
model_path = "weather_model.h5"
scaler_path = "scaler.pkl"
secret = "aaaa54121"

os.makedirs(os.path.dirname(file_name), exist_ok=True)

model = None
scaler = None


class WeatherRecord(BaseModel):
    Date: date
    Location: str
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: Optional[float] = None
    Sunshine: Optional[float] = None
    WindGustDir: WindDirection
    WindGustSpeed: float
    WindDir9am: WindDirection
    WindDir3pm: WindDirection
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float = Field(ge=0, le=100)
    Humidity3pm: float = Field(ge=0, le=100)
    Pressure9am: float
    Pressure3pm: float
    Cloud9am: float = Field(ge=0, le=9)
    Cloud3pm: float = Field(ge=0, le=9)
    Temp9am: float
    Temp3pm: float
    RainToday: YesNo
    RainTomorrow: Optional[YesNo] = None

    class Config:
        schema_extra = {
            "example": {
                "Date": "2023-01-01",
                "Location": "Sydney",
                "MinTemp": 15.0,
                "MaxTemp": 25.0,
                "Rainfall": 0.0,
                "Evaporation": 4.2,
                "Sunshine": 8.5,
                "WindGustDir": "SE",
                "WindGustSpeed": 35.0,
                "WindDir9am": "E",
                "WindDir3pm": "SE",
                "WindSpeed9am": 10.0,
                "WindSpeed3pm": 15.0,
                "Humidity9am": 75.0,
                "Humidity3pm": 60.0,
                "Pressure9am": 1015.0,
                "Pressure3pm": 1013.0,
                "Cloud9am": 4.0,
                "Cloud3pm": 6.0,
                "Temp9am": 18.0,
                "Temp3pm": 23.0,
                "RainToday": "No",
                "RainTomorrow": "No"
            }
        }


def preprocess_data(df: pd.DataFrame):
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Location", "Date"])

    df = df.dropna(subset=["RainTomorrow"])

    df = df.set_index(["Location", "Date"])
    df = df.groupby("Location").ffill().reset_index()

    # Convert Yes/No to 1/0
    df["RainToday"] = df["RainToday"].map({"No": 0, "Yes": 1})
    df["RainTomorrow"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})

    # Drop remaining RainToday/RainTomorrow NaNs
    df = df.dropna(subset=["RainToday", "RainTomorrow"])

    # Convert to integers
    df["RainToday"] = df["RainToday"].astype(int)
    df["RainTomorrow"] = df["RainTomorrow"].astype(int)

    # Handle other missing values (median imputation)
    cols_to_fill = ["Humidity3pm", "Pressure9am", "Cloud3pm", "WindGustSpeed"]
    for col in cols_to_fill:
        df[col] = df.groupby("Location")[col].transform(
            lambda x: x.fillna(x.median()))

    # Final cleanup
    df = df.dropna()
    return df


def create_sequences(df: pd.DataFrame):
    features = [
        "Rainfall",
        "Humidity3pm",
        "Pressure9am",
        "RainToday",
        "Cloud3pm",
        "WindGustSpeed",
        "Temp3pm",
    ]
    target = "RainTomorrow"
    sequence_length = 7
    X, y = [], []

    for location, group in df.groupby("Location"):
        group = group.sort_values("Date")
        group_data = group[features].values
        target_data = group[target].values

        for i in range(len(group_data) - sequence_length):
            X.append(group_data[i: i + sequence_length])
            y.append(target_data[i + sequence_length])

    return np.array(X), np.array(y).astype(int), features


def build_model(input_shape):
    model = Sequential(
        [
            LSTM(
                128,
                input_shape=input_shape,
                return_sequences=True,
                recurrent_dropout=0.2,
            ),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, recurrent_dropout=0.2),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


@app.get("/")
async def root():
    return {"message": "Weather Prediction API - a"}


@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    try:
        with open(file_name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "Data file updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_model(sec: str):
    if sec != secret:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        df = pd.read_csv(file_name)
        df = preprocess_data(df)

        X, y, features = create_sequences(df)

        global scaler
        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        scaler = StandardScaler()
        X_flat_scaled = scaler.fit_transform(X_flat)
        X_scaled = X_flat_scaled.reshape(n_samples, n_timesteps, n_features)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        global model
        model = build_model((n_timesteps, n_features))
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
        )
        class_weight_dict = dict(enumerate(class_weights))

        history = model.fit(
            X_train,
            y_train,
            epochs=30,
            batch_size=128,
            validation_split=0.2,
            class_weight=class_weight_dict,
            callbacks=[early_stop],
            verbose=1,
        )

        model.save(model_path)
        import joblib

        joblib.dump(scaler, scaler_path)

        test_results = model.evaluate(X_test, y_test, verbose=0)
        return {
            "message": "Model trained successfully",
            "test_accuracy": float(test_results[1]),
            "test_precision": float(test_results[2]),
            "test_recall": float(test_results[3]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(location: str, last_n_days: int = 7):
    try:
        global model, scaler

        if model is None:
            if not os.path.exists(model_path):
                raise HTTPException(
                    status_code=400, detail="Model not trained yet")
            model = load_model(model_path)

        if scaler is None:
            if not os.path.exists(scaler_path):
                raise HTTPException(status_code=400, detail="Scaler not found")
            import joblib

            scaler = joblib.load(scaler_path)

        df = pd.read_csv(file_name)
        df = preprocess_data(df)

        loc_df = df[df["Location"] == location]
        if len(loc_df) == 0:
            raise HTTPException(
                status_code=400, detail=f"Location '{location}' not found in data"
            )

        loc_df = loc_df.sort_values("Date")

        features = [
            "Rainfall",
            "Humidity3pm",
            "Pressure9am",
            "RainToday",
            "Cloud3pm",
            "WindGustSpeed",
            "Temp3pm",
        ]

        last_dates = loc_df["Date"].tail(last_n_days).tolist()
        last_features = loc_df[features].tail(last_n_days).values

        scaled_features = scaler.transform(
            last_features.reshape(-1, len(features)))
        scaled_sequence = scaled_features.reshape(
            1, last_n_days, len(features))

        predictions = []
        confidence = []
        current_sequence = scaled_sequence.copy()

        for _ in range(7):
            pred_prob = float(model.predict(current_sequence, verbose=0)[0][0])
            pred = 1 if pred_prob >= 0.5 else 0

            new_day = current_sequence[0, -1].copy()
            new_day[3] = pred

            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = new_day

            predictions.append(pred)
            confidence.append(pred_prob)

        last_date = pd.to_datetime(last_dates[-1])
        forecast_dates = [
            (last_date + datetime.timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(7)
        ]

        return {
            "location": location,
            "forecast": [
                {"date": date, "rain_predicted": bool(
                    pred), "confidence": conf}
                for date, pred, conf in zip(forecast_dates, predictions, confidence)
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/addRec")
async def add_record(record: WeatherRecord):
    try:
        df = pd.read_csv(file_name)

        record_dict = record.dict()

        if record_dict["MaxTemp"] < record_dict["MinTemp"]:
            raise HTTPException(
                status_code=400,
                detail="MaxTemp cannot be less than MinTemp"
            )

        if record_dict["Temp3pm"] > record_dict["MaxTemp"] or record_dict["Temp9am"] < record_dict["MinTemp"]:
            raise HTTPException(
                status_code=400,
                detail="Temperature readings must be within MinTemp and MaxTemp range"
            )

        record_dict["Date"] = record_dict["Date"].strftime("%Y-%m-%d")

        df = pd.concat([df, pd.DataFrame([record_dict])], ignore_index=True)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Location", "Date"])

        df.to_csv(file_name, index=False)

        return {
            "message": "Record added successfully",
            "record": record_dict
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
