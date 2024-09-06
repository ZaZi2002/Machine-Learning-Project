# Machine Learning Project: Dimensionality Reduction and Ensemble Learning

## Project Overview
This project is for Introduction to Machine Learning course and applies **dimensionality reduction** techniques, specifically **Principal Component Analysis (PCA)**, and uses **ensemble learning** methods such as **Random Forests** and **Decision Trees**. The goal is to improve the classification performance on a dataset by optimizing the number of features and weak learners. Key metrics such as accuracy, precision, recall, F1-score, and AUPRC (Area Under Precision-Recall Curve) are used to evaluate the model's performance.

### Key Features:
- **Dimensionality Reduction with PCA:** Reducing the number of features by maintaining the most variance-rich components.
- **Ensemble Learning:** Using multiple weak learners (Decision Trees) with both hard and soft voting strategies to enhance prediction accuracy.
- **Performance Metrics:** The model's output is evaluated using accuracy, precision, recall, F1-score, and AUPRC.

### Steps in the Notebook:
1. **Data Preprocessing:** 
   - Mean normalization and zero-centering of the data.
   - PCA to reduce the dimensionality of the dataset based on explained variance.
   
2. **Model Training:**
   - Train and test the model using the **Random Forest** estimator.
   - Implement ensemble learning with different numbers of weak learners.
   
3. **Performance Evaluation:**
   - Calculate key metrics including accuracy, precision, recall, F1-score, and AUPRC for both PCA-reduced data and ensemble learners.
   
4. **Optimization:** 
   - The number of PCA components and weak learners is optimized to balance performance and computational cost.

### Metrics Achieved:
- **Accuracy:** 97.7%
- **Precision:** 98.4%
- **Recall:** 98.7%
- **F1-Score:** 98.6%
- **AUPRC:** 98.1%

## Requirements
To run the project, you need the following Python libraries:
- `numpy`
- `scikit-learn`
- `matplotlib`
