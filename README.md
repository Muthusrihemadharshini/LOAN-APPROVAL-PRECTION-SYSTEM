
# 🏦 Loan Default Prediction System

## 📌 Overview

A machine learning-based web application that predicts the risk of loan default based on applicant financial and demographic details. The system is designed to simulate real-world credit risk assessment by combining predictive modeling with practical decision logic.

---

## 🚀 Features

* 📊 Predicts loan default risk (Low / Medium / High)
* 🧠 Uses machine learning for risk evaluation
* ⚖️ Handles imbalanced data using SMOTE
* 🎯 Optimized using precision, recall, and threshold tuning
* 🌐 Interactive web app built with Streamlit
* 💡 Incorporates business rules for realistic decision-making

---

## 🛠️ Tech Stack

* **Python**
* **Scikit-learn**
* **Random Forest Classifier**
* **SMOTE (Imbalanced-learn)**
* **Streamlit**
* **NumPy, Pandas**

---

## 🧠 Machine Learning Approach

* Performed data preprocessing and feature engineering
* Created new features like:

  * Income-Credit Ratio
  * EMI Ratio
* Addressed class imbalance using SMOTE
* Trained multiple models (Logistic Regression, XGBoost, Random Forest)
* Selected Random Forest based on balanced performance
* Tuned model using:

  * Class weights
  * Hyperparameters
  * Threshold adjustment

---

## 📊 Model Performance

* Achieved ~88–91% accuracy
* Improved detection of high-risk applicants
* Reduced misclassification using optimized thresholds
* Evaluated using:

  * Confusion Matrix
  * Precision & Recall

---

## 💡 Key Insight

High accuracy alone is not sufficient for imbalanced datasets. This project focuses on improving real-world performance by prioritizing the detection of risky applicants over naive accuracy optimization.

---

## 🌐 Application

The deployed web app allows users to:

* Input financial details
* Receive instant risk prediction
* View probability-based decision output

---

## 📁 Project Structure

```
├── app.py
├── model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── notebook.ipynb
```

---

## ⚡ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 Future Improvements

* Add model explainability (feature importance / SHAP)
* Improve UI/UX design
* Deploy using cloud platforms
* Integrate real-world datasets/APIs

---

## 🙌 Conclusion

This project demonstrates the end-to-end development of a machine learning system—from data preprocessing and model training to deployment—while addressing real-world challenges like class imbalance and model evaluation.

