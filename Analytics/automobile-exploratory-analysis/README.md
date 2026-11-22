# Automobile Pricing and Exploratory Analysis

## What this project is about

**Question:** What characteristics of a vehicle appear to be associated with its price, and how should the underlying data be prepared before drawing conclusions?

I used Python to explore an automobile dataset containing vehicle prices, engine specifications, fuel efficiency and body styles. The project is a practical example of using raw operational data to understand product differences and the factors that may influence pricing. It is an **exploratory analysis**, not a car-price prediction system.

## What I implemented

- Inspected the dataset and removed duplicate rows and records with missing values.
- Converted price, horsepower and engine measurements into numeric variables.
- Examined price distributions, the relationship between engine size and price, and the number of vehicles represented by each manufacturer.
- Produced a correlation heatmap to compare numerical characteristics.

## Technology and methods

Python, pandas, NumPy, Matplotlib and seaborn; data cleaning, descriptive statistics, visualisation and correlation analysis.

## How to run

1. Install the packages listed in `requirements.txt`.
2. Supply an appropriately licensed `automobile.txt` dataset in the project folder.
3. Open `automobile.ipynb` in Jupyter and run the cells in order.

## What this demonstrates

The ability to clean imperfect data, investigate possible drivers of product pricing and explain findings visually. Correlation alone does not establish causation.

**Data:** The notebook uses an automobile dataset, which should be placed alongside the notebook as `automobile.txt`.
