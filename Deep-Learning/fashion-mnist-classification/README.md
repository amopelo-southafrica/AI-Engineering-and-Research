# Fashion-MNIST Clothing Image Classification

## What this project is about

**Question:** Can a neural network learn to recognise the type of clothing shown in a small greyscale image?

I built a PyTorch image classifier using the Fashion-MNIST benchmark, which contains labelled pictures of clothing and footwear. The model predicts one of ten clothing categories from the image pixels. The project covers supervised model design, training and evaluation for image recognition.

## What I implemented

- Loaded the Fashion-MNIST training and test sets and normalised image values.
- Built a multilayer perceptron with 784 input features, two hidden layers of 128 and 64 units, and ten output classes.
- Set up cross-entropy loss, the Adam optimiser and a ten-epoch training loop.
- Evaluated predictions using accuracy, macro precision, macro recall, macro F1 and a confusion matrix.

## Technology and methods

Python, PyTorch, torchvision, scikit-learn and NumPy; neural networks, gradient-based optimisation and classification metrics.

## How to run

1. Install the Python packages from `requirements.txt` and Jupyter.
2. Open `fashion_mnist.ipynb`. The notebook downloads Fashion-MNIST on first use.
3. **Known correction needed in the submitted notebook:** change `import numpy as n` to `import numpy as np` before executing the random-seed cell.
4. Run the cells from the top.

## What this demonstrates

The end-to-end steps involved in training an image classifier and evaluating its errors rather than relying on accuracy alone.

The notebook calculates accuracy, precision, recall, F1 and a confusion matrix when run.
