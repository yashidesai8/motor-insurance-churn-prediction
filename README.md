# Predicting Customer Churn in Motor Insurance: A Machine Learning Approach to Customer Retention

## Project Overview

This repository contains the code, dataset, and supporting materials for my MSc Business Analytics dissertation project at the University of Greenwich.

The aim of this project is to develop and evaluate machine learning models for predicting customer churn in the motor insurance industry and to identify the key factors influencing customer retention.

## Research Questions

### RQ1

Which machine learning model provides the best predictive performance for customer churn prediction under class imbalance conditions?

### RQ2

Which customer, policy, and vehicle-related factors are the strongest predictors of churn, and how can these insights support customer retention strategies?

## Dataset

The dataset consists of motor insurance policy records containing customer, policy, vehicle, and claims-related information.

After data cleaning and preprocessing, the dataset was used to train and evaluate multiple machine learning models for churn prediction.

## Methodology

The project follows a structured machine learning workflow:

1. Data Cleaning and Preprocessing
2. Feature Engineering
3. Exploratory Data Analysis
4. Data Standardisation
5. Model Development and Hyperparameter Tuning
6. Model Evaluation
7. Feature Importance Analysis
8. Customer Segmentation using K-Means Clustering

## Machine Learning Models

The following classification models were evaluated:

* Logistic Regression
* K-Nearest Neighbours (KNN)
* Naïve Bayes
* Support Vector Classifier (SVC)
* Random Forest

## Key Findings

* Random Forest achieved the best overall predictive performance.
* Renewal timing, policy tenure, seniority, premium, and claims history were identified as the most influential churn predictors.
* Customer relationship factors were found to have greater predictive importance than vehicle-related characteristics.
* Customer segmentation identified a high-risk customer group suitable for targeted retention strategies.

## Repository Contents

* `motor_insurance_churn.ipynb` – Complete analysis notebook
* `dataset.csv` – Dataset used for analysis
* `README.md` – Project documentation

## Tools and Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Author

Yashi Desai

MSc Business Analytics

University of Greenwich
