
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

1. [Exploratory Data Analysis (EDA)](#eda)
2. [Outliers Detection (IQR, Z-score) & Treatment](#outliers)
3. [Splitting the Dataset](#splitting)
4. [Feature Scaling](#scaling)
5. [Coarse-to-Fine Hyperparameter Tuning Search](#tuning)
6. [Models Training](#training)
    - 6.1. [Naive Bayes](#bayes)
    - 6.2. [Support Vector Machine (SVM)](#svm)
    - 6.3. [K-nearest Neighbors (KNN)](#knn)
7. [Validation Report](#report)
8. [Conclusions](#conclusions)

**Note**: From this point on, a brief summary of each section has been provided into this document. You can view the full code with all the details and comments here:📙[Project 1 - Jupyter Notebook](https://github.com/mjimenezj/Portfolio/blob/main/Projects/Project_1/Project_1.ipynb)

## 1. Exploratory Data Analysis (EDA) <a id="eda"></a>

The exploratory analysis reveals that there are 21 explanatory variables. All are quantitative, with either discrete or continuous values. The target variable 'Fetal_state' is binary, with 0 indicating a normal state and 1 indicating an abnormal state of the fetus. Its distribution is as follows:

<p align="center"> <img src="Images/fetal_state.png" alt="Imagen" width="300" /> </p>

> The distribution is **imbalanced**, so it's important to keep this in mind when splitting the dataset to ensure a fair representation of both classes. 

The Pearson correlation coefficient has been calculated for the numerical variables of the dataset. The Pearson correlation coefficient measures the strength and direction of the linear relationship between two numerical variables ranging from -1 to 1. Correlation Matrix: 

<p align="center"> <img src="Images/corr.png" alt="Imagen" width="800" /> </p>

> The 3 explanatory variables with the highest correlation to `Fetal_state` are **`ASTV`** (Pearson Coef. = 0.49): percentage of time with abnormal short-term variability; **`ALTV`** (Pearson Coef. = 0.48): percentage of time with abnormal long-term variability and **`AC`** (Pearson Coef. = -0.37): number of accelerations per second, counted in discrete units. 
> On the other hand, it is worth noting that there are several strong correlations between independent variables.

## 2. Outliers detection (IQR, Z-score) & Treatement <a id="outliers"></a>

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

## 3. Splitting the Dataset <a id="splitting"></a>

Before training the algorithms, it is necessary to split the data into Training and Test sets. 

The following split has been chosen:
- Training set size = **70%**
- Test set size = **30%**

Additionally, **stratified sampling** has been applied to ensure that the target variable, which is imbalanced (as detected in the EDA), is proportionally represented in both the training and test sets.

```
# Splitting into explanatory variables (X) and Fetal_state (y)
X = data_transformed.drop('Fetal_state', axis=1)
y = data_transformed['Fetal_state']

# Splitting into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, stratify = y, random_state = 0)
```

## 4. Feature scaling <a id="scaling"></a>

Feature scaling in machine learning is a crucial step in data pre-processing before building a model. Proper scaling can significantly impact the performance of a model, making the difference between a weak and a strong one. Since KNN and SVM are based on the distances between points, it will be essential to scale the variables so that all data is within the same range. To achieve this, the decision was made to remove the mean and set the standard deviation to 1, meaning that the Z-score has been calculated through the `StandardScaler()` from sklearn:

*You can read more about different scalers in https://scikit-learn.org/stable/modules/preprocessing.html*

> Note: **Data leakage** occurs when information from the test set or future data influences the model during training, leading to unrealistic results and biased model evaluation. This happens when the model has access to data it shouldn't, such as performing preprocessing (e.g., feature scaling) before splitting the dataset, or including features that are improperly correlated with the target.

> Avoiding data leakage is crucial for obtaining an accurate evaluation of the model's performance on unseen data and **this is why Feature Scaling should be made after splitting the dataset**

```
# The scaler object is defined
scaler = StandardScaler().fit(X_train)

# Transformation
standardized_X_train = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)
```


## 5. Coarse-to-Fine Hyperparameter Tuning Search <a id="tuning"></a>

Using `RandomizedSearchCV` followed by `GridSearchCV` methods from sklearn [(learn more)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html) is an effective strategy for hyperparameter optimization with a "coarse to fine" search approach:

1. **First, RandomizedSearchCV is used**:
   - RandomizedSearchCV explores a wide range of hyperparameter values by testing random combinations within a defined space.
   - This helps perform the **coarse search**, quickly covering a large search space.

2. **Then, GridSearchCV is used**:
   - Once promising ranges of hyperparameters are identified, **GridSearchCV** performs an exhaustive search over those narrowed-down values.
   - This is ideal for **fine search**, ensuring precise tuning of hyperparameters.

This combination maximizes efficiency and helps in finding the optimal model configuration.


<p align="center"> <img src="Images/grid_random.png" alt="Imagen" width="400" /> </p>

> Image Source: Bergstra, J., Bengio, Y.: Random search for hyper-parameter optimization. Journal of Machine Learning Research 13, 281–305 (2012)


In addition, during the Hyperparameter search, stratified cross-validation is used with `StratifiedKFold()`  [(link)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html). Stratified K-Fold is a variation of K-Fold cross-validation that ensures each fold of the dataset has the same proportion of each target class as the entire dataset. This is particularly useful when dealing with imbalanced datasets (as this case), as it prevents the model from being biased toward overrepresented classes during training.

## 6. Models Training <a id="training"></a>

### 6.1. **Naive Bayes** <a id="bayes"></a>

Naive Bayes is a classification algorithm based on **Bayes' Theorem**, which is used to predict the probability of different classes based on the values of input features. It is called "naive" because it assumes that the features are independent of each other, which is a simplifying assumption that doesn't always hold in real-world data. Despite this assumption, Naive Bayes often performs surprisingly well.

Hyperparameters (Gaussian Naive Bayes):

- `var_smoothing`:
  - This parameter adds a small value to the variance of each feature to avoid division by zero or very small numbers in the Gaussian probability formula. 
  - Typical values: values in the range of `1e-9` to `1e0`.

```
# Best params
gnb = GaussianNB(var_smoothing = 0.2)
```
**Results**

<p align="center"> <img src="Images/nb.png" alt="Imagen" /> </p>



```
============ Metrics for NB ============
ROC-AUC Training Set: 	0.9445296523517384
ROC-AUC Test Set: 	0.9670040485829958
PR-AUC Training Set: 	0.8258362343983097
PR-AUC Test Set: 	0.8926223531912151
F1-score Training Set: 	0.7411003236245954
F1-score Test Set: 	0.8145454545454546
----------
Accuracy Test Set: 	0.919558359621451
Precision Test Set: 	0.8296296296296296
Recall Test Set: 	0.8
Specificity Test Set: 	0.9534412955465587
Trainig time (seconds): 0.003999233245849609
```
  
