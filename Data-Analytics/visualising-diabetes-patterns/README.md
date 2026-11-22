# Diabetes Outcome Data Visualisation

This project reviews an original HyperionDev exploratory notebook using an **outcome-labelled diabetes dataset**, with attention to plotting, missing-value proxies and the limits of cross-sectional associations.

## Files and lineage

- [`diabetes_visualisation_corrected.ipynb`](diabetes_visualisation_corrected.ipynb): separately repaired notebook with zero-aware median handling and a proper final markdown interpretation.
- [`diabetes_visualisation_original.ipynb`](diabetes_visualisation_original.ipynb): original exploratory notebook. It expects a file named `diabetes(1).csv`.
- [`diabetes(1).csv`](diabetes(1).csv): **restored matching dataset**, 768 rows, with `Glucose`, `BMI`, `Age`, `Outcome` and other fields. Copied exactly from the user's original HyperionDev M10T03 Random Forests task; source Git blob SHA `db6f317685a4668bc0529b16ea1544022e780002`.
- [`diabetes.csv`](diabetes.csv): **different 442-row diabetes-progression regression dataset**, retained because it was present in the original M08T09 task materials. It is **not compatible** with this notebook's `Outcome` and `Glucose` code.
- [`diabetes-report.pdf`](diabetes-report.pdf): preserved coursework report, not independently validated against the corrected input.
- [Research paper](RESEARCH_PAPER.md) and [PDF](RESEARCH_PAPER.pdf): source-based audit, reproduced descriptive results, assumptions and limitations.

## Running the notebook

Install pandas, seaborn, Matplotlib and Jupyter. Keep `diabetes(1).csv` beside the notebook and run the cells sequentially. The notebook replaces zero values in five measurements with **medians calculated while zeros are still included**. This is a methodological limitation, particularly for `Insulin`, and is discussed in the research paper.

The data contain an educational outcome label. Associations should **not** be presented as clinical diagnosis, causal inference or general population prevalence.
