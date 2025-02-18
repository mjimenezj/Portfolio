
# 👶🏻🩺 Fetal Health Prediction During Childbirth

## Context

Reducing child mortality is a crucial goal outlined in several of the United Nations' Sustainable Development Goals (SDGs) and serves as a vital indicator of human development.

The UN aims to eliminate preventable deaths of newborns and children under 5 years of age by 2030. In parallel to the issue of child mortality, maternal mortality also remains a pressing concern.

In this context, Cardiotocograms (CTGs) offer a simple and cost-effective way to assess fetal health, providing healthcare professionals with vital data to help prevent both maternal and child mortality. The equipment works by emitting ultrasound pulses and analyzing the returned signals, providing insights into fetal heart rate (FHR), fetal movements, uterine contractions, and more.

A dataset of 2,126 CTGs has been collected, along with their respective diagnostic features. These CTGs were classified by three expert obstetricians, and a consensus classification label was assigned to each. The diagnostic features represent the set of variables that will be used to estimate the fetal condition (target) as either normal or abnormal.

## 🎯 Objective

To predict the fetal health status during childbirth using Supervised Classification Algorithms.

## Dataset Used

This project uses data published by **Ayres-de-Campos, *et al.*, (2000). *Journal of Maternal-Fetal Medicine*, 9(5), 311-318.**

The data is available [here](https://www.tandfonline.com/doi/abs/10.3109/14767050009053454) and [here](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification).

## Machine Learning Algorithms

- Naive Bayes
- Support Vector Machine (SVM)
- K-nearest neighbors (KNN)

## Table of Contents 

1. Exploratory Data Analysis (EDA)
2. Outliers detection (IQR, Z-score) & Treatement.
3. Splitting the Dataset
4. Feature scaling
5. Coarse-to-Fine Hyperparameter Tuning Search.
6. Models Training. <br>
7. Validation Report: ROC-AUC, Accuracy, F1-score, PR-AUC, etc.
8. Conclusion

**Note**: From this point on, a brief summary of each section has been provided into this document. You can view the full code with all the details and comments here:📘 📘

## 1. Exploratory Data Analysis (EDA)








