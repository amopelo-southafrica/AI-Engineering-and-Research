# Finding Groups in the Iris Dataset with Hierarchical Clustering

## What this project is about

**Question:** Can a computer group similar flowers from their measurements without being told which species they belong to?

I explored unsupervised learning using the Iris benchmark dataset. Rather than train a model on labelled species, I used measured petal length and width to discover natural groupings and examine how different definitions of distance affect the result.

## What I implemented

- Loaded the Iris dataset from scikit-learn and selected petal length and petal width.
- Standardised the selected measurements before clustering.
- Compared dendrograms using single and complete linkage with Euclidean and city-block (Manhattan) distance.
- Applied agglomerative hierarchical clustering with three groups.
- Plotted the resulting clusters and calculated a silhouette score to assess separation.

## Technology and methods

Python, pandas, scikit-learn, SciPy and Matplotlib; unsupervised learning, feature standardisation, hierarchical clustering and silhouette analysis.

## How to run

Install `requirements.txt` and open `hierarchical_clustering.ipynb` in Jupyter. The Iris dataset is loaded through scikit-learn, so no external CSV is required.

## What this demonstrates

Choosing and comparing clustering methods, interpreting dendrograms and evaluating whether discovered groups are distinct.

The notebook calculates and displays the silhouette score when run.
