# 🧠 MNIST CNN Classifier with Deep Diagnostics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/solidvv/mnist-classification/blob/main/mnist_full_analysis.ipynb)

An end-to-end Computer Vision project implementing a Convolutional Neural Network (CNN) for handwritten digit classification (MNIST). 

Unlike standard tutorials, this project focuses on **production-ready practices** and **model interpretability**, featuring automated checkpointing, GPU acceleration, and a comprehensive suite of diagnostic tools to analyze model errors.

**Performance:** ~98.6% Accuracy on Test Set

## Key Features

* **Robust Architecture:** Custom CNN implementation using PyTorch.
* **Smart Training Loop:**  **GPU/CUDA Support:** Auto-detects hardware for accelerated training.
    * **Model Checkpointing:** Automatically saves the *best* model weights (based on validation accuracy), not just the final epoch.
    * **Live Monitoring:** Real-time plotting of Loss and Accuracy during training.
* **Advanced Diagnostics:**
    * **Confusion Matrix:** Visualizes class-wise confusion (e.g., 4 vs 9).
    * **Classification Report:** Precision, Recall, and F1-Score metrics for every digit.
    * **Misclassification Analysis:** Automatically detects, extracts, and visualizes specific samples where the model failed, providing insight into edge cases.

## Project Structure

```text
├── checkpoints                # Saved model weights (best_model.pth)
├── MNIST/                     # Dataset storage (auto-downloaded)
├── model.py                   # CNN Architecture definition
├── train.py                   # Training and evaluation scripts
├── mnist_full_analysis.ipynb  # Main Notebook: Training, Inference & Visualizations
├── requirements.txt           # Project dependencies

└── README.md                  # Project documentation

