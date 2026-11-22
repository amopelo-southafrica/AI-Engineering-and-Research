# Titanic Random Forest

## Research question and approach

I trained a random forest classifier on Titanic passenger records, examined feature importance and compared different tree counts and maximum depths.

## Implementation

I used pandas, numpy, matplotlib, scikit-learn. My code and analysis are available in [titanic-random-forest.ipynb](titanic-random-forest.ipynb).

## Reproduce the project

Install Jupyter and the listed Python libraries. Place `titanic (2).csv` in the notebook directory. Run the cells in order.

## Implementation notes

The notebook compares hyperparameters on its test split. Those comparisons are exploratory, not a separate unbiased final evaluation. The source dataset is not packaged in this repository.
