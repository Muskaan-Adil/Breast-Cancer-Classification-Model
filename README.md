# Cancer Diagnosis – Classification Model

## Project Overview

This project involves applying **Logistic Regression** to predict the diagnosis of breast cancer based on various medical features. The primary goal is to build a model that can classify tumors as malignant or benign and evaluate the model’s performance using metrics like accuracy and F1 score.

The project highlights the importance of preprocessing, feature selection, and model evaluation for building an effective and reliable predictive model for medical applications.

---

## Key Features

* **Data Inspection**: Loaded and explored the dataset, checked for missing values, and preprocessed the data.
* **Preprocessing**: Encoded categorical features and scaled numerical features to improve model performance.
* **Logistic Regression**: Applied Logistic Regression to predict cancer diagnoses based on the input features.
* **Model Evaluation**: Evaluated model performance using accuracy, F1 score, and confusion matrix.
* **Feature Importance**: Analyzed and displayed the importance of each feature in the prediction using coefficients.
* **Visualization**: Created visual summaries of model performance and feature importance.

---

## Dataset Information

**Source**: [Cancer Wisconsin Dataset](https://drive.google.com/file/d/1TQDAoNFJ7DtsneYGt83nrd7HEnQOtCjB/view).

**Columns Include**:

* `id`: Unique identifier for each record (dropped during preprocessing).
* `diagnosis`: Malignant or Benign tumor diagnosis (target variable).
* Various medical attributes such as:

  * `radius_mean`: Mean radius of the tumor.
  * `texture_mean`: Mean texture of the tumor.
  * `perimeter_mean`: Mean perimeter of the tumor.
  * `area_mean`: Mean area of the tumor.
  * `smoothness_mean`: Mean smoothness of the tumor.
  * and several others.

---

## Feature Engineering and Preprocessing

### Loading and Initial Processing

* Loaded the dataset using `pandas` and explored it with `df.head()`, `df.describe()`, and `df.info()`.
* Dropped the `id` and `Unnamed: 32` columns.
* Encoded the target variable `diagnosis` as 0 (Benign) and 1 (Malignant).
* Checked and handled missing values (if any).
* Split the data into training and testing sets using `train_test_split`.

### Scaling and Model Training

* Scaled the numerical features using `StandardScaler` to ensure they are on the same scale for Logistic Regression.
* Trained a **Logistic Regression** model to predict tumor diagnosis.

---

## Model Building and Evaluation

The model was trained and evaluated using the following metrics:

* **Accuracy** – Percentage of correctly classified instances.
* **F1 Score** – Balance between precision and recall, especially for imbalanced classes.
* **Confusion Matrix** – Breakdown of true positives, true negatives, false positives, and false negatives.
* **Classification Report** – Precision, recall, and F1 scores for both classes.

Additionally, feature importance was determined based on the coefficients of the Logistic Regression model.

---

## Visualizations

Created:

* **Confusion Matrix** plot to visually represent classification results.
* **Bar plot** for the importance of top features, showing which features are most predictive of cancer diagnosis.

---

## Report

For a comprehensive explanation of each step, including preprocessing, model results, and visual insights, refer to the full analysis in **[Breast\_Cancer\_Report.md](Breast_Cancer_Report.md)**.
