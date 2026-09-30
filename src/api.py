from fastapi import FastAPI
from src.train import MODEL_PATH
import joblib
from pydantic import BaseModel
import pandas as pd


app = FastAPI(title= "House prediction API")

@app.get("/")
def root():
    return {"message": "The API is working!"}

@app.get("/health")
def health():
    return {"status": "ok"}

# We load the model once the API is opened, not every time that and endpoind is called

pipeline = joblib.load(MODEL_PATH)

# We create a class in order to create objects that API is going to recieve
class House(BaseModel):
    OverallQual: int #Overall calification must be an integer  
    GrLivArea: float
    Neighborhood: str
    TotalBsmtSF: float
    GarageArea: float
    OverallCond: int
    LotArea: float
    YearBuilt: int
    YearRemodAdd: int
    FullBath: int #Number of full equiped bathrooms
    OpenPorchSF: float

#we create a Post endpoint, so the customer can send data to the API
@app.post("/predict")
def predict(house:House):
    data = pd.DataFrame([house.model_dump()])
    prediction = pipeline.predict(data)[0]
    return {"predicted_price": round(float(prediction),2)}