# 💳 Credit Card Fraud Detection using XGBoost

This project focuses on building a machine learning model to detect credit card fraud using the **XGBoost** algorithm. The challenge lies in the **highly imbalanced dataset**, where fraudulent transactions make up only **0.17%** of all records.

## 📁 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
- **Instances**: 284,807 transactions
- **Fraud Cases**: 492 (~0.17%)

## 🧠 Project Objective

- Handle imbalanced data effectively without using sampling techniques
- Build an accurate fraud detection model using **XGBoost**
- Evaluate performance using appropriate classification metrics

## ⚙️ Steps Covered

- Data loading and exploratory analysis
- Feature scaling using `RobustScaler` for `Amount` and `Time`
- Correlation analysis and feature selection
- Train-test split with stratification
- Handling class imbalance using `scale_pos_weight` in XGBoost
- Model training and evaluation

## 📊 Model Performance

| Metric            | Score |
| ----------------- | ----- |
| AUC Score         | 96.6% |
| F1-Score (fraud)  | 86%   |
| Precision (fraud) | 90%   |
| Recall (fraud)    | 82%   |
| Accuracy          | ~100% |

These results show that **XGBoost**, when properly configured for class imbalance, can effectively detect rare fraudulent transactions **without needing oversampling or undersampling**.

## 🛠 Technologies Used

- Python
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn
- XGBoost

## 🧪 Possible Future Improvements

- Hyperparameter tuning using GridSearchCV or Optuna
- Try ensemble models combining XGBoost with other classifiers
- Apply SMOTE or ADASYN for comparison
- Use SHAP values for model interpretability
- Deploy the model for real-time fraud detection

## 📌 Note

Due to file size limitations, the dataset is not included in this repository. Please download it from the [Kaggle dataset link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) and place it in the `data/` folder.
