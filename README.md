# Mortality Prediction for In-Hospital ICU Patients

## Overview
This project implements machine learning models to predict in-hospital mortality for ICU patients using:

- Artificial Neural Networks (ANN)
- XGBoost (Extreme Gradient Boosting)

The project utilizes data from the **PhysioNet Challenge 2012**, focusing on the first 48 hours of ICU admission, to evaluate and enhance predictive accuracy in critical care settings.

## Table of Contents

1. [Features](#features)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Models](#models)
5. [Results](#results)
6. [Limitations and Future Work](#limitations-and-future-work)
7. [Installation](#installation)
8. [Usage](#usage)

## Features

- **Preprocessing:** Handling missing data, standardization, and oversampling using SMOTE.
- **Model Implementation:**
  - ANN: Captures non-linear relationships in ICU data.
  - XGBoost: Gradient-boosted trees for high interpretability and precision.
- **Performance Metrics:** Accuracy, Precision, Recall, F1-Score, and AUC-ROC.
- **Feature Importance Visualization:** Key features influencing mortality predictions.

## Dataset

- **Source:** [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/)
- **Size:** 12,000+ ICU admissions.
- **Features:**
  - Demographic: Age, gender, ICU type, initial weight.
  - Physiological: Vital signs, lab metrics (time series).
  - Outcome: Binary target variable (`In-hospital_death`).
- **Challenges:** Missing data, time-series processing, and class imbalance.

## Preprocessing

- **Handling Missing Values:**
  - Mean imputation for physiological metrics.
  - Median imputation for height and weight.
- **Scaling:** Standardized continuous variables for uniform feature scaling.
- **Encoding:** Label encoding for categorical variables.
- **Addressing Class Imbalance:** Applied SMOTE to ensure the model focuses on minority class predictions.

## Models

### Artificial Neural Network (ANN)

- **Architecture:**
  - Input Layer: 22 preprocessed features.
  - Hidden Layers: 5 layers with ReLU activation.
  - Output Layer: Sigmoid activation for binary classification.
- **Training Parameters:**
  - Epochs: 100, Batch Size: 32.
  - Optimizer: Adam, Loss Function: Binary Cross-Entropy.

### XGBoost Classifier

- **Parameters:**
  - n_estimators: 100, Learning Rate: 0.1.
  - Max Depth: 5, Regularization: L1 and L2.
- **Feature Importance:** Highlights critical features influencing predictions.

## Results

| Model         | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------------|----------|-----------|--------|----------|---------|
| **ANN**       | 93%      | 88%       | 95%    | 90%      | xx%     |
| **XGBoost**   | 95%      | 100%      | 92%    | 96%      | xx%     |

- **Observations:**
  - XGBoost outperformed in precision and interpretability.
  - ANN excelled in recall and sensitivity for critical healthcare use cases.

## Limitations and Future Work

### Limitations
- Smaller dataset size limits generalizability.
- Handling missing values may introduce bias.
- ANN and XGBoost models lack intuitive interpretability for clinical applications.

### Future Work
- Incorporate real-time ICU data streams for dynamic predictions.
- Develop interpretable models with attention mechanisms.
- Address demographic biases and ensure fairness in predictions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mortality-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mortality-prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place the dataset in the `data/` directory.
2. Run the Jupyter Notebook for preprocessing, model training, and evaluation:
   ```bash
   jupyter notebook "Mortality Prediction.ipynb"
   ```

## Contributing

Contributions are welcome! Please fork this repository and create a pull request for your changes.


