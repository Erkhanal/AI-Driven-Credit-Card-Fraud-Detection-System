# Credit Card Fraud Detection System

## Table of Contents:
* [Introduction](#introduction)
* [Problem Statement](#problem-statement)
* [Objectives](#objectives)
* [Analysis Approach](#analysis-approach) 
* [Final Analysis](#final-analysis)
* [Technologies Used](#technologies-used)
* [References](#references)
* [Contact](#contact)

## Introduction:  
<div align="justify"> This repository contains the code for Credit Card Fraud Detection System. Fraudulent credit card transactions pose a major risk to financial institutions, leading to financial losses, decreased institution’s trust, and reduced customer satisfaction. With the exponential rise in digital payment channels, detecting fraud has become an important priority for banks and financial institutions.</div><br>

<div align="justify"> Machine learning offers a powerful solution to detect and prevent fraudulent transactions. By automating fraud detection processes, banks can reduce the need for manual reviews, avoid costly chargebacks, prevent legitimate transaction denials, and enhance the overall customer experience.</div><br>

<div align="justify"> This capstone project explores the use of machine learning models to predict fraudulent transactions in credit cards. By analyzing a publicly available dataset from a collaboration between Worldline and the Machine Learning Group, we aim to build a robust fraud detection model capable of identifying fraudulent activities with high precision.</div>

## Problem Statement:  
<div align="justify">  For many banks, customer retention and profitability are key priorities. Fraudulent credit card transactions present a major challenge to this objective, as they lead to significant financial losses, erosion of customer trust, and damage to brand reputation.</div><br>

<div align="justify"> According to the Nilson report, global banking fraud was estimated to reach $30 billion by 2020. The rise of digital payment platforms has significantly increased the opportunities for fraudsters to exploit weaknesses in the banking system. Therefore, banks and financial institutions must implement proactive and efficient fraud detection mechanisms.</div>

## Objectives:  
<div align="justify"> The main objective of this project is to build machine learning models that will help banks and financial institutions to detect fraud with greater accuracy. For this we will use various machine learning techniques and evaluate the model performance.</div>

## Analysis Approach:    
<div align="justify"> To tackle this problem effectively, I have established a structured data analysis approach.</div><br>

1. Reading and Understanding the Data<br>
2. Data Preprocessing<br>
3. Exploratory Data Analytics (EDA)<br>
4. Train/Test Split<br>
5. Model Building and Hyperparameter Tuning<br>
6. Model Evaluation and Selection

### 1. Reading and Understanding the Data:
<div align="justify">It includes data loading and analyzing to understand the data structure, types of variables, target variable distribution, understanding the nature of the features and their relevance to the problem, identifying any patterns or trends that might help in predicting fraudulent transactions.</div><br>

### 2. Data Preprocessing:
<div align="justify">It includes checking data type, cleaning, and preparing the data for analysis, handling missing values, encoding categorical variables, dropping unnecessary columns, and standardizing data.</div><br>

### 3. Exploratory Data Analytics (EDA):
EDA will be performed to uncover deeper insights and relationships within the data. Mainly the following analyses will be conducted:<br>

**Univariate Analysis:** Understanding the distribution of individual features.<br>

**Bivariate Analysis:** Analyzing relationships between pairs of features and the target variable.<br>

**Imbalanced Data Check:** Identifying the imbalance between fraudulent and non-fraudulent transactions.<br>

**Skewness Check:** Checking for any skewed distributions that might affect model performance.<br>

### 4. Train/Test Split:
<div align="justify">After completing the EDA, dataset will be split into training and testing sets. To ensure the model performs well on unseen data, stratified train-test splitting will be applied. This helps maintain the proportion of fraudulent to non-fraudulent transactions in both the training and testing datasets.</div><br>

<div align="justify">Since the dataset is highly imbalanced, specific resampling techniques such as Random Over Sampling, SMOTE (Synthetic Minority Over-sampling Technique), and ADASYN (Adaptive Synthetic Sampling) are applied to balance the dataset.</div><br>

### 5. Model Building and Hyperparameter Tuning:
<div align="justify">In this phase where we will experiment with different machine learning models and adjust their hyperparameters to find the most accurate model for predicting fraudulent transactions. </div><br>

<div align="justify">Various machine learning models are trained on both imbalanced and balanced datasets, using various resampling techniques (Random Over Sampling, SMOTE, and ADASYN) to compare their performance. Hyperparameter tuning is conducted to optimize the models, using methods such as grid search and random search, in conjunction with stratified k-fold cross-validation.</div><br>

### 6. Model Evaluation and Model Selection:
<div align="justify">It includes evaluating, validating, and comparing the performance of various machine learning models and analyzing which model provides better insights into fraud transactions. The best-performing model will be selected based on the evaluation metrics such as precision, recall, and F1-score, including the ROC-AUC score and the confusion matrix. Key considerations include:</div><br>

**Model Accuracy:** How well the model classifies fraudulent transactions.<br>

**Model Stability:** Consistent performance across multiple folds of cross-validation.<br>

**Feature Importance:** Identifying the most influential features for predicting fraud.</div>

## Final Analysis:
<div align="justify">  In summary, instead of focusing solely on overall accuracy across the entire dataset, our priority was to maximize the detection of fraud cases (recall) while keeping the associated costs in check (precision). We applied XGBoost on SMOTE-augmented data and achieved the best evaluation metrics.<br>

## Conclusion:
<div align="justify"> This project demonstrates the power of machine learning in identifying fraudulent credit card transactions. By carefully handling data imbalance and testing multiple models with extensive hyperparameter tuning, we developed a robust fraud detection system that can significantly improve fraud prevention for banks. The final model provides a highly accurate solution that can be implemented by banks and financial institutions to reduce fraud, protect customer trust, and minimize financial losses. </div><br>

<div align="justify"> In a real-world application, this system can be further enhanced by integrating real-time transaction data and employing additional feature engineering techniques and further fine-tuning. Machine learning-based fraud detection systems like this are crucial for the financial sector to mitigate losses and maintain customer trust in an era of increasing digital transactions.</div>

## Technologies Used:
- Python, version 3 
- NumPy for numerical computations
- Matplotlib and seaborn for data visualization
- Pandas for data manipulation
- Statsmodels for statistical modeling
- Sklearn for machine learning tasks
- Jupyter Notebook for interactive analysis

## References:
- Python documentations
- ML/AI documentations
- Stack Overflow
- Kaggle
- Medium AI Articles

## Contact:
Created by: https://github.com/Erkhanal - feel free to contact!
