
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
3. [Splitting the Dataset](#splitting-the-dataset)
4. Feature scaling
5. Coarse-to-Fine Hyperparameter Tuning Search.
6. Models Training. <br>
7. Validation Report: ROC-AUC, Accuracy, F1-score, PR-AUC, etc.
8. Conclusion

**Note**: From this point on, a brief summary of each section has been provided into this document. You can view the full code with all the details and comments here:📙[Project 1 - Jupyter Notebook](https://github.com/mjimenezj/Portfolio/blob/main/Projects/Project_1/Project_1.ipynb)

## 1. Exploratory Data Analysis (EDA)

The exploratory analysis reveals that there are 21 explanatory variables. All are quantitative, with either discrete or continuous values. The target variable 'Fetal_state' is binary, with 0 indicating a normal state and 1 indicating an abnormal state of the fetus. Its distribution is as follows:

<p align="center"> <img src="Images/fetal_state.png" alt="Imagen" width="300" /> </p>

> The distribution is **imbalanced**, so it's important to keep this in mind when splitting the dataset to ensure a fair representation of both classes. 

The Pearson correlation coefficient has been calculated for the numerical variables of the dataset. The Pearson correlation coefficient measures the strength and direction of the linear relationship between two numerical variables ranging from -1 to 1. Correlation Matrix: 

<p align="center"> <img src="Images/corr.png" alt="Imagen" width="800" /> </p>

> The 3 explanatory variables with the highest correlation to `Fetal_state` are **`ASTV`** (Pearson Coef. = 0.49): percentage of time with abnormal short-term variability; **`ALTV`** (Pearson Coef. = 0.48): percentage of time with abnormal long-term variability and **`AC`** (Pearson Coef. = -0.37): number of accelerations per second, counted in discrete units. 
> On the other hand, it is worth noting that there are several strong correlations between independent variables.

## 2. Outliers detection (IQR, Z-score) & Treatement.

Outliers were detected with two methods:

- **IQR**: or interquartile range. The interquartile range is the central 50% of the data. If the data is divided into 4 equal portions or quartiles (Q1-Q4), the two central quartiles represent the IQR, i.e., IQR = Q3−Q1. This method considers any data points outside of 1.5 times the IQR from Q1 or Q3 as outliers, i.e., $[Q1−1.5×IQR, Q3+1.5×IQR]$.

- **Z-score**: This calculates how many standard deviations each value is from the mean. In other words, the mean will be at 0, the standard deviation at ± 1, and the values will be distributed in a range from +infinity to -infinity, being positive if greater than the mean and negative if less than the mean. It is common to use a threshold of ± 3 to detect outliers.

**Feature Transformation**: after outliers detection, some variables were transfored by one of the following methods:
- Logarithmic transformation
- Root square
- Binning
- Winsorization

This is a comparison before vs after data transformation. The threshold are the vertival red lines (Z-score = +-3):

<p align="center"> <img src="Images/outliers.png" alt="Imagen"/> </p>

> The data transformation have drastically reduced the number of outliers present in the data.

## Splitting the Dataset

hola hola




hola
