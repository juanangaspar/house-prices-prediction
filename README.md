# House price prediction

Data analysis and machine learning project focused on predicting house prices using Python.

## Project goal

The goal of this project is to build a machine learning model capable of predicting the sale price of a house based on its features.

The project follows a data science workflow: data loading, cleaning, exploratory analysis, feature preparation, model training, evaluation and interpretation of feature importance.

## Dataset

The project uses a house prices dataset taken from Kaggle, with numerical and categorical variables related to the characteristics of the properties.

## Technologies used

- Python
- pandas
- NumPy
- matplotlib
- scikit-learn
- Jupyter Notebook

## Project timeline

1. Data loading and initial exploration.
2. Data cleaning and preparation.
3. Exploratory data analysis.
4. Modeling.
5. Feature importance analysis.
6. Conclusions.

## Models used

- Linear regression
- Random Forest Regressor

## Results

A linear regression model was first used as a baseline model. This model showed many limitations, especially for high-priced houses and in the presence of large errors.

A Random Forest Regressor model was then trained, which obtained better results, reducing large errors and improving the R² value. In addition, this model made it possible to analyze the importance of the variables to better understand which features have the greatest influence on the price prediction.

## Conclusions

The Random Forest model showed better performance than linear regression for this regression problem.

The project demonstrates a complete initial machine learning workflow, including data cleaning, exploratory analysis, model comparison and interpretation of results.

## Possible future improvements

- Apply cross-validation.
- Optimize hyperparameters.
- Handle outliers in more detail.
- Try more advanced models such as Gradient Boosting or XGBoost.
- Improve feature engineering.
