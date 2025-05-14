# Detailed Report – Breast Cancer Detection using Classification Model

---

## 1. **Data Overview**

The project focused on predicting whether a breast tumor is **malignant (1)** or **benign (0)** using the **Breast Cancer Wisconsin dataset**. The dataset was first loaded and explored to understand its structure:

* **Shape**: The dataset contains several diagnostic features for 569 samples.
* **Preview**: `df.head()` was used to inspect the first few rows.
* **Summary Stats**: `df.describe()` revealed ranges and distributions of features.
* **Data Types**: `df.info()` showed numeric types and confirmed the presence of two non-essential columns (`id`, `Unnamed: 32`) which were dropped.
* The target variable `diagnosis` was mapped to binary values: `M → 1`, `B → 0`.

### Missing Values

A check confirmed **no missing values** remained after dropping irrelevant columns.

---

## 2. **Exploratory Data Analysis**

A **count plot** visualized class distribution:

* The dataset was moderately imbalanced with more benign cases than malignant.
* This imbalance was handled by **stratified splitting** during train-test separation.

---

## 3. **Preprocessing Steps**

* **Feature Scaling**: StandardScaler was applied to normalize the data.
* **Train-Test Split**: 80/20 split using stratification to preserve class proportions.
* **Feature Matrix** `X`: All columns except `diagnosis`.
* **Target Vector** `y`: The encoded `diagnosis` column.

---

## 4. **Model Training**

A **Logistic Regression model** was trained with:

* **Maximum Iterations**: 10,000 (to ensure convergence)
* **Random State**: 42 (for reproducibility)

---

## 5. **Model Evaluation**

The model’s predictions were evaluated on the test set using multiple metrics:

### Key Metrics:

* **Accuracy**: 0.9649

* **F1 Score**: 0.9512

* **Confusion Matrix**:

  |                  | Predicted Benign | Predicted Malignant |
  | ---------------- | ---------------- | ------------------- |
  | Actual Benign    | 71               | 1                   |
  | Actual Malignant | 3                | 39                  |

* **Classification Report**:

  | Class         | Precision | Recall | F1-score | Support |
  | ------------- | --------- | ------ | -------- | ------- |
  | Benign (0)    | 0.96      | 0.99   | 0.97     | 72      |
  | Malignant (1) | 0.97      | 0.93   | 0.95     | 42      |

* **ROC AUC Score**: Area Under the Curve was plotted to evaluate true/false positive trade-offs.

---

## 6. **Visualization Summary**

* **Confusion Matrix**: Displayed as a heatmap for easy interpretation.
* **Top 10 Feature Importances**: Plotted based on absolute coefficient values from the logistic model.
* **ROC Curve**: Showed high model separability with an AUC close to 1.

---

## 7. **Top 10 Most Influential Features**

| Feature              | Coefficient (abs. value sorted) |
| -------------------- | ------------------------------- |
| worst perimeter      | 1st                             |
| mean concave points  | 2nd                             |
| worst concave points | 3rd                             |
| mean perimeter       | 4th                             |
| worst radius         | 5th                             |
| mean concavity       | 6th                             |
| mean radius          | 7th                             |
| worst area           | 8th                             |
| radius error         | 9th                             |
| mean area            | 10th                            |

These features had the strongest influence on prediction outcomes and aligned well with domain knowledge.

---

## Conclusion

This project successfully demonstrates the use of **Logistic Regression for binary classification** of breast tumors using well-preprocessed features.

### Achievements:

* Achieved **96% accuracy** with strong F1 and recall scores.
* Identified key predictive features influencing diagnosis.
* Developed a complete workflow from raw data to model evaluation with visual insights.

### Future Directions:

* Use advanced models like **Random Forest**, **XGBoost**, or **SVM**.
* Perform **hyperparameter tuning** to further optimize results.
* Address class imbalance using **SMOTE** or **class weights**.
* Explore **dimensionality reduction** techniques like PCA.
