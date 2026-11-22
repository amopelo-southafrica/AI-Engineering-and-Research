# Predicting Customer Churn for a Telecommunications Business

## What this project is about

**Business question:** Can customer account and service information help identify subscribers who are likely to leave?

I used a telecommunications customer dataset to build and compare two churn prediction models. For a subscription business, identifying likely cancellations can help teams investigate retention risks and decide where to focus customer-support efforts. The project focuses on classification methods that can support analysis of customer retention risk.

## What I implemented

- Prepared customer data by converting `TotalCharges` to a number, removing incomplete rows, excluding customer IDs and encoding the churn outcome.
- Applied feature preparation, including Min-Max scaling, before fitting classification models.
- Trained logistic regression and random forest classifiers.
- Compared predictions with accuracy, confusion matrices, precision and recall, with an emphasis on missed churners (false negatives).
- Calculated the random forest's out-of-bag error as an additional diagnostic.

## Technology and methods

Python, pandas, NumPy, scikit-learn, Matplotlib and seaborn; classification, feature preprocessing, model evaluation and error analysis.

## How to run

Install the packages in `requirements.txt`, obtain the appropriately licensed `Telco-Customer-Churn.csv` dataset, and run `customer_churn_prediction.ipynb` in Jupyter.

## What this demonstrates

How to turn customer records into a predictive modelling question, compare a simpler interpretable model with an ensemble model, and consider the cost of false-negative predictions.

**Data:** Supply `Telco-Customer-Churn.csv` to reproduce the model comparison. Evaluation metrics are calculated in the notebook.
