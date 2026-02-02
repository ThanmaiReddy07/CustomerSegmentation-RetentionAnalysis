# Customer Segmentation & Retention Analysis

## 📌 Overview
This project analyzes telecom customer data to:
- Perform **customer segmentation** using clustering (KMeans, Agglomerative).
- Build and evaluate multiple **churn prediction models** (Logistic Regression, Random Forest, XGBoost, Gradient Boosting).
- Select the best model based on **F1 Score** and save it for deployment.
- Provide a **Streamlit app** for interactive churn prediction (single customer or batch CSV upload).

The goal is to help businesses identify customers likely to churn and design retention strategies.

---

## ⚙️ Features
- **Data Preprocessing**: Cleaning, encoding categorical variables, handling imbalance with SMOTE-ENN.
- **Segmentation**: KMeans and Agglomerative clustering to group customers.
- **Model Building**: Hyperparameter tuning with RandomizedSearchCV.
- **Model Comparison**: Accuracy, Precision, Recall, F1 Score.
- **Deployment**: Streamlit app for online and batch predictions.
- **Visualization**: Interactive charts for churn distribution and segmentation.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Libraries**:
  - pandas, numpy, scikit-learn
  - imbalanced-learn (SMOTE-ENN)
  - xgboost
  - matplotlib, seaborn, plotly
  - streamlit
  - joblib

---


---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ThanmaiReddy07/CustomerSegmentation-RetentionAnalysis.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the streamlit app:
   ```bash
   streamlit run app.py

---   
  
## 👩‍💻 Author
Developed by Thanmai Reddy Chinnapareddy  
Computer Science Student | AI & Data Science Enthusiast

---   

