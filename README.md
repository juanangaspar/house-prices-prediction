# House Price Prediction API

End-to-end machine learning project to predict house prices: from exploratory data analysis and model selection to a REST API that serves predictions.

## Project overview

- **Exploratory analysis and modeling** (notebooks folder): data cleaning, missing value treatment, and comparison between a linear regression and a Random Forest Regressor.
- **Feature selection for deployment**: the final model uses 11 variables instead of the 79 available. Variables were selected based on their importance in the Random Forest, avoiding highly correlated (redundant) variables and prioritizing data that a user can easily provide. The reduced model achieves performance similar to the full model (see the notebook for the full comparison).
- **Training script** (src/train.py): builds a scikit-learn Pipeline that combines preprocessing (one-hot encoding of the neighborhood) and the model, trains it, evaluates it and saves it to the models folder. Packaging preprocessing and model together guarantees that new data is transformed exactly as during training.
- **REST API** (src/api.py): built with FastAPI. It loads the trained pipeline and serves predictions through a /predict endpoint, with automatic input validation.

## Results

Performance of the deployed model (Random Forest with 11 variables) on the test set (20% of the data):

| Metric | Value |
|---|---|
| MAE | ~18,550 $ |
| RMSE | ~32,860 $ |
| R² | 0.844 |

## Project structure

```
house-prices-prediction/
├── data/
│   └── train.csv              # Training data
├── notebooks/
│   └── house_price_prediction.ipynb   # EDA, modeling and feature selection
├── src/
│   ├── train.py               # Trains and saves the model pipeline
│   └── api.py                 # FastAPI application
├── models/                    # Trained model (generated, not versioned)
├── requirements.txt
└── README.md
```

## How to run locally

Requirements: Python 3.12 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/juanangaspar/house-prices-prediction.git
cd house-prices-prediction

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (creates models/model.joblib)
python src/train.py

# 5. Start the API
uvicorn src.api:app --reload
```

Once the server is running, open **http://127.0.0.1:8000/docs** in your browser. This interactive documentation page lets you try every endpoint without writing any code.

## Using the API

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Welcome message |
| GET | /health | Health check, returns a status "ok" |
| POST | /predict | Returns the predicted price of a house |

### About the data and units

The model was trained with the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset from Kaggle, which contains residential sales in **Ames, Iowa (USA)** between 2006 and 2010.

Since the data comes from the United States, **all areas are expressed in square feet** (1 ft² ≈ 0.093 m²) and **prices in US dollars**. Entering values in square meters will produce wrong predictions.

### Input fields

| Field | Description | Unit / range |
|---|---|---|
| OverallQual | Overall quality of materials and finish | 1 (very poor) to 10 (excellent) |
| GrLivArea | Above-ground living area | ft² |
| TotalBsmtSF | Total basement area | ft² |
| GarageArea | Garage area | ft² |
| OverallCond | Overall condition of the house | 1 (very poor) to 10 (excellent) |
| LotArea | Lot size | ft² |
| YearBuilt | Year of construction | year |
| YearRemodAdd | Year of the last remodel (same as YearBuilt if never remodeled) | year |
| FullBath | Full bathrooms above ground | count |
| OpenPorchSF | Open porch area | ft² |
| Neighborhood | Neighborhood code within Ames | see list below |

Valid neighborhood codes: Blmngtn, Blueste, BrDale, BrkSide, ClearCr, CollgCr, Crawfor, Edwards, Gilbert, IDOTRR, MeadowV, Mitchel, NAmes, NPkVill, NWAmes, NoRidge, NridgHt, OldTown, SWISU, Sawyer, SawyerW, Somerst, StoneBr, Timber, Veenker. The full description of every variable is available in the data_description.txt file provided by Kaggle.

### Example request

POST request to /predict with the following JSON body:

```json
{
  "OverallQual": 7,
  "GrLivArea": 1700,
  "TotalBsmtSF": 1000,
  "GarageArea": 480,
  "OverallCond": 5,
  "LotArea": 9000,
  "YearBuilt": 2000,
  "YearRemodAdd": 2005,
  "FullBath": 2,
  "OpenPorchSF": 40,
  "Neighborhood": "CollgCr"
}
```

Response:

```json
{
  "predicted_price": 189013.6
}
```

## Project status

**This project is a work in progress.** The model and the API are fully functional locally. The final goal is to package the application in a **Docker** container and **deploy it to the cloud**, so that the API is publicly accessible.

Next steps:

- Stricter input validation (value ranges and valid neighborhoods) and response schemas
- Automated tests with pytest
- Docker containerization
- Cloud deployment
- CI pipeline with GitHub Actions
- k-fold cross-validation for a more robust model comparison