from fastapi import FastAPI
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

# Create FastAPI instance
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
