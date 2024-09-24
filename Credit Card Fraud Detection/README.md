# Fraud Detection with Machine Learning Models

## Project Overview
This project focuses on the detection of fraudulent transactions using machine learning techniques. Given the highly imbalanced nature of the fraud detection problem, where fraudulent cases are rare compared to normal transactions, we apply multiple models and evaluate their performance in identifying fraudulent activity.

## Models Used
Three machine learning models were implemented and evaluated on the dataset:
1. **Isolation Forest**: An unsupervised learning algorithm commonly used for anomaly detection.
2. **SGD Classifier (Stochastic Gradient Descent)**: A linear classifier optimized for large-scale datasets.
3. **Neural Network**: A deep learning model trained to classify transactions as fraudulent or non-fraudulent.

## Performance Metrics
To evaluate the models' performance, we used the following metrics, which are important for imbalanced classification problems:
- **Precision**: Measures how many of the predicted frauds were actual frauds.
- **Recall**: Measures how many of the actual frauds were correctly identified.
- **F1-Score**: The harmonic mean of precision and recall, providing a balanced evaluation.
- **AUC (Area Under the ROC Curve)**: Evaluates the model’s ability to distinguish between classes.

## Results
### 1. Isolation Forest
- **Precision (Fraud)**: 0.23
- **Recall (Fraud)**: 0.12
- **F1 Score (Fraud)**: 0.16
- **AUC**: 0.56

The Isolation Forest showed poor performance due to its low recall and F1 score. It was able to identify very few fraudulent transactions.

### 2. SGD Classifier
- **Precision (Fraud)**: 0.89
- **Recall (Fraud)**: 0.46
- **F1 Score (Fraud)**: 0.60
- **AUC**: N/A

The SGD Classifier performed better, achieving a higher precision and F1 score. However, its recall was still moderate, missing a significant number of fraud cases.

### 3. Neural Network
- **Precision (Fraud)**: 0.89
- **Recall (Fraud)**: 0.76
- **F1 Score (Fraud)**: 0.82
- **AUC**: N/A

The Neural Network outperformed both the Isolation Forest and SGD Classifier, especially in terms of recall and F1 score. It provided the best balance between identifying fraudulent cases and avoiding false positives.

## Key Files
- `main.ipynb`: Jupyter Notebook containing the code for model training, evaluation, and visualizations.
- `roc_curve.png`: A plot comparing the ROC curves of the different models.

## How to Run the Project
1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/RajaaLebnaiti/fraud-detection-project.git
    ```
2. Install the required dependencies:
   
3. Open the Jupyter notebook:
    ```bash
    jupyter notebook main.ipynb
    ```
4. Run all cells to train the models and evaluate the results.

## Dataset
The dataset contains two classes:
- `0`: Non-fraudulent transactions
- `1`: Fraudulent transactions

Given the imbalance in the data, additional techniques such as oversampling, undersampling, and model tuning can further enhance performance.

## Recommendations
Based on the results, the Neural Network model is the most effective in detecting fraudulent transactions. For further improvements, consider:
- Hyperparameter tuning of the Neural Network (e.g., learning rates, network architecture).
- Using techniques like oversampling, undersampling, or class weighting to handle class imbalance.
- Investigating additional models such as Random Forest or XGBoost for fraud detection.

## Conclusion
This project demonstrates the application of machine learning algorithms to detect fraud in a highly imbalanced dataset. The results suggest that deep learning models, such as Neural Networks, can provide significant improvements in identifying fraudulent transactions when compared to traditional methods like Isolation Forests.


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==================================================
# Nom du Projet : Credit Card Fraud Detection
# Fichier : main.ipynb
# Auteur : Rajaa LEBNAITI
# Email : rajaa.lebnaiti@gmail;com

# ==================================================

# Copyright 2024 
# ==================================================