# Handwritten Digit Recognition

I applied a random forest classifier to optical handwritten-digit samples, then inspected confusion-matrix errors and macro-averaged accuracy, precision, recall and F1 score.

## Tools and implementation

I used pandas, numpy, scikit-learn in [my notebook](optical_digit_classification.ipynb).

## Reproduce my work

Install the listed packages, open Jupyter and run the notebook with `optdigits.tes` (supply this dataset separately; it was not included in the source folder).

The classification metrics are computed from the held-out portion of the supplied dataset.

HyperionDev task M10T04.
