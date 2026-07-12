import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import joblib

# Input format
class Email(BaseModel):
    message: str

# ANN architecture
class ANN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.out(x))
        return x

# Load pipeline
pipeline = joblib.load("spam_pipeline.pkl")

vectorizer = pipeline["tfidf"]
input_size = pipeline["input_size"]

model = ANN(input_size)

model.load_state_dict(
    pipeline["model_state"]
)

model.eval()

# FastAPI
app = FastAPI(title="Spam Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "Spam Detection API Running"
    }

@app.post("/predict")
def predict(email: Email):

    vector = vectorizer.transform(
        [email.message]
    ).toarray()

    tensor = torch.tensor(
        vector,
        dtype=torch.float32
    )

    with torch.no_grad():
        result = model(tensor)

    probability = result.item()

    prediction = "spam" if probability >= 0.5 else "ham"

    return {
        "prediction": prediction,
        "confidence": round(probability, 4)
    }