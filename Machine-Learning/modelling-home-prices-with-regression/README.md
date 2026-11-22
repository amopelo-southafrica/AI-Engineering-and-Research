# Predicting House Sale Prices with Linear Regression

## What this project is about

**Question:** How much information about a property's sale price can be explained using its living area and garage size?

I used the Ames, Iowa housing dataset to build a multiple linear regression model. The project demonstrates a clear and interpretable approach to valuation: identify measurable property characteristics, estimate their relationship with sale price, and examine how the predictions compare with actual outcomes.

## What I implemented

- Inspected the dataset, checked for missing values and selected two explanatory variables: `Gr_Liv_Area` and `Garage_Area`.
- Examined property sale-price distributions and plotted each feature against sale price.
- Split the observations into training (75%) and test (25%) subsets.
- Trained a scikit-learn `LinearRegression` model to estimate `Sale_Price`.
- Calculated mean squared error and root mean squared error, visualised predicted versus actual prices, and inspected model coefficients.

## Technology and methods

Python, pandas, NumPy, scikit-learn, Matplotlib and seaborn; multiple regression, train/test evaluation and model interpretation.

## How to run

Install the listed dependencies, place an appropriately licensed `ames.csv` file beside the notebook, and run `ames_housing_regression.ipynb` in Jupyter.

## What this demonstrates

Applying mathematical modelling to a practical valuation question and interpreting both coefficients and prediction errors. This particular model focuses on two property features: living area and garage area.

**Data:** Supply `ames.csv` to reproduce the analysis. The notebook calculates RMSE and shows predicted versus actual sale prices.
