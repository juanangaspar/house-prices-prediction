"""Train the model and save it in models / """

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT /"models" / "model.joblib"

NUMERIC_VARIABLES = ["OverallQual","GrLivArea", "TotalBsmtSF", "GarageArea", "OverallCond", "LotArea", "YearBuilt", "YearRemodAdd", "FullBath", "OpenPorchSF"]
CATEGORICAL_VARIABLES = ["Neighborhood"]

TARGET = "SalePrice"

def build_pipeline():
    #Preprocessing 
    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", NUMERIC_VARIABLES),
        ("cat", OneHotEncoder(handle_unknown= "ignore"), CATEGORICAL_VARIABLES)])

    #Create the pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=0))
    ])
    return pipeline

def main():
    # Read the data
    df = pd.read_csv(DATA_PATH)

    # Split target variable from the other
    x = df[NUMERIC_VARIABLES + CATEGORICAL_VARIABLES]
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Train the model using the pipeline
    pipeline = build_pipeline()
    pipeline.fit(x_train,y_train)

    # Evaluate the model with the test data and print metrics
    y_pred = pipeline.predict(x_test)

    mae = mean_absolute_error(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r_2 = r2_score(y_test,y_pred)


    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R^2:",r_2)

    #We save the model trained
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
    
