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
from typing import List, Dict
import os

# Create FastAPI instance
app = FastAPI()

file_name = "data.csv"
model_path = "weather_model.h5"
scaler_path = "scaler.pkl"

# Global variables to store model and scaler
model = None
scaler = None


def preprocess_data(df: pd.DataFrame):
    # Convert and sort dates
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Location", "Date"])

    # Initial cleanup: Drop rows with missing RainTomorrow
    df = df.dropna(subset=["RainTomorrow"])

    # Forward-fill within locations
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
        df[col] = df.groupby("Location")[col].transform(lambda x: x.fillna(x.median()))

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
            X.append(group_data[i : i + sequence_length])
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
    return {"message": "Weather Prediction API"}


@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        with open(file_name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "Data file updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_model():
    try:
        # Load and preprocess data
        df = pd.read_csv(file_name)
        df = preprocess_data(df)

        # Create sequences
        X, y, features = create_sequences(df)

        # Scale features
        global scaler
        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        scaler = StandardScaler()
        X_flat_scaled = scaler.fit_transform(X_flat)
        X_scaled = X_flat_scaled.reshape(n_samples, n_timesteps, n_features)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Build and train model
        global model
        model = build_model((n_timesteps, n_features))
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
        )
        class_weight_dict = dict(enumerate(class_weights))

        # Train
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

        # Save model and scaler
        model.save(model_path)
        import joblib

        joblib.dump(scaler, scaler_path)

        # Evaluate
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

        # Load model and scaler if not loaded
        if model is None:
            if not os.path.exists(model_path):
                raise HTTPException(status_code=400, detail="Model not trained yet")
            model = load_model(model_path)

        if scaler is None:
            if not os.path.exists(scaler_path):
                raise HTTPException(status_code=400, detail="Scaler not found")
            import joblib

            scaler = joblib.load(scaler_path)

        # Load and preprocess data
        df = pd.read_csv(file_name)
        df = preprocess_data(df)

        # Get location data
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

        # Get last available dates
        last_dates = loc_df["Date"].tail(last_n_days).tolist()
        last_features = loc_df[features].tail(last_n_days).values

        # Scale features
        scaled_features = scaler.transform(last_features.reshape(-1, len(features)))
        scaled_sequence = scaled_features.reshape(1, last_n_days, len(features))

        predictions = []
        confidence = []
        current_sequence = scaled_sequence.copy()

        for _ in range(7):
            pred_prob = float(model.predict(current_sequence, verbose=0)[0][0])
            pred = 1 if pred_prob >= 0.5 else 0

            # Create new day data
            new_day = current_sequence[0, -1].copy()
            new_day[3] = pred  # Update RainToday with prediction

            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = new_day

            predictions.append(pred)
            confidence.append(pred_prob)

        # Generate forecast dates
        last_date = pd.to_datetime(last_dates[-1])
        forecast_dates = [
            (last_date + datetime.timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(7)
        ]

        return {
            "location": location,
            "forecast": [
                {"date": date, "rain_predicted": bool(pred), "confidence": conf}
                for date, pred, conf in zip(forecast_dates, predictions, confidence)
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
