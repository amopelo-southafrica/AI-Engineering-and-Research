# Iris Logistic Regression

## Research question and approach

I trained a logistic regression classifier to distinguish Iris setosa from other Iris species using four flower measurements. I split the data into training and test sets and calculated classification metrics.

## Implementation

I used pandas, numpy, scikit-learn. My code and analysis are available in [iris-logistic-regression.ipynb](iris-logistic-regression.ipynb).

## Reproduce the project

Install Jupyter and the listed Python libraries. Place `Iris.csv` in the notebook directory. Run the cells in order.

## Implementation notes

The positive/negative matrix indexing in the source notebook is nonstandard; inspect the label mapping before interpreting precision and recall. The source dataset is not packaged in this repository.
