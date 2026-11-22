# US Arrests: PCA and Clustering

## What I developed

I standardized state-level arrest measurements, used PCA to reduce their dimensions and compared K-means with hierarchical clustering of the transformed records.

## Technical approach

I used pandas, numpy, seaborn, matplotlib, scikit-learn, scipy, and documented the working code in [us-arrests-pca-and-clustering.ipynb](us-arrests-pca-and-clustering.ipynb).

## Reproduce my work

Install Jupyter and the listed libraries. Supply `UsArrests.csv` in the working directory. Open the notebook and run the cells in order.

## Notes

I plotted state labels and investigated explained variance. The analysis describes clustering of the supplied dataset, not underlying causes.
