
# 👶🏻🩺 Fetal Health Classification

## Context

Reducing child mortality is a crucial goal outlined in several of the United Nations' Sustainable Development Goals (SDGs) and serves as a vital indicator of human development.

The UN aims to eliminate preventable deaths of newborns and children under 5 years of age by 2030. In parallel to the issue of child mortality, maternal mortality also remains a pressing concern.

In this context, Cardiotocograms (CTGs) offer a simple and cost-effective way to assess fetal health, providing healthcare professionals with vital data to help prevent both maternal and child mortality. The equipment works by emitting ultrasound pulses and analyzing the returned signals, providing insights into fetal heart rate (FHR), fetal movements, uterine contractions, and more.

A dataset of 2,126 CTGs has been collected, along with their respective diagnostic features. These CTGs were classified by three expert obstetricians, and a consensus classification label was assigned to each. The diagnostic features represent the set of variables that will be used to estimate the fetal condition (target) as either normal or abnormal.

## 🎯 Objective

To predict the fetal health status during childbirth using Supervised Classification Algorithms.

## Dataset Used

This project uses data publicly available at: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

## Machine Learning Algorithms
- Naive Bayes
- Support Vector Machine (SVM)
- K-nearest neighbor (KNN)

## Table of Contents 


1. Exploratory Data Analysis (EDA)
2. Outliers detection (IQR, Z-score) & Feature Scaling.
3. Stratified Data Split.
4. Coarse-to-Fine Hyperparameter Tuning Search.
5. Models Training. <br>
6. Validation Report: ROC-AUC, Accuracy, F1-score, PR-AUC, etc.



## 1. Exploratory Data Analysis (EDA)


