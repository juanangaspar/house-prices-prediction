from fastapi import FastAPI
from src.train import MODEL_PATH
import joblib
from pydantic import BaseModel, Field, model_validator
import pandas as pd
from typing import Literal


app = FastAPI(title= "House prediction API")

@app.get("/")
def root():
    return {"message": "The API is working!"}

@app.get("/health")
def health():
    return {"status": "ok"}

# We load the model once the API is opened, not every time that and endpoind is called

pipeline = joblib.load(MODEL_PATH)

# We create a class in order to create objects that API is going to receive
# Also we aim to make the API robust so it does not accept impossible data, like a GrLivArea of -100
class House(BaseModel):
    OverallQual: int = Field(ge=1, le=10, description="Overall quality of materials and finish (1-10)")
    GrLivArea: float = Field(ge=300, description="Above-ground living area, in square feet, with a minimum of 300")
    Neighborhood: Literal[
        "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
    "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
    "NAmes", "NPkVill", "NWAmes", "NoRidge", "NridgHt", "OldTown",
    "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
    ] = Field(description="Neighborhood codes from Ames, where data of the dataset was taken")
    TotalBsmtSF: float = Field(ge=0, description="Surface of the basement, in square feet")
    GarageArea: float = Field(ge=0, description="Surface of the garage, in square feet")
    OverallCond: int = Field(ge=1, le=10, description="Overall condition of the house at the moment of sale")
    LotArea: float = Field(gt=0, description="Surface of the whole property")
    YearBuilt: int = Field(ge=1800, le=2010, description="Year which the house was built. It has to be prior to 2010 since model was trained with houses built between 1800 and 2010")
    YearRemodAdd: int = Field(ge=1800, le=2010, description="Year of the last remodel (same as YearBuilt if never remodeled)")
    FullBath: int = Field(ge=0, description="Number of fully equiped bathrooms in the house")
    OpenPorchSF: float = Field(ge=0, description="Surface of the porch of the house")

    @model_validator(mode="after")
    def check_remodel_year(self):
        if self.YearRemodAdd<self.YearBuilt:
            raise ValueError("YearRemodAdd cannot be before YearBuilt")
        return self

# We also are going to create a format for answers of the API
class PredictionResponse(BaseModel):
    predicted_price: float = Field(description="Predicted price, in US Dollars")

#We create a Post endpoint, so the customer can send data to the API
@app.post("/predict")
def predict(house:House) -> PredictionResponse:
    data = pd.DataFrame([house.model_dump()])
    prediction = pipeline.predict(data)[0]
    return PredictionResponse(predicted_price=round(float(prediction),2))


