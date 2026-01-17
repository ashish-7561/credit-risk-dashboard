# Enterprise Credit Risk Assessment System

# Live Demo: https://credit-risk-dashboard-7wxv.onrender.com

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/AI-Gradient%20Boosting-orange?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Frontend-Gradio-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Deployed-success?style=for-the-badge)

# Project Overview
The Credit Risk Assessment System is an advanced Machine Learning application designed to help financial institutions automate loan eligibility decisions. By analyzing applicant demographics and financial history, the AI predicts the likelihood of loan default with high accuracy.

*Business Goal: Minimize financial loss by identifying high-risk borrowers while streamlining the approval process for creditworthy applicants.

#  Key Features
*  Advanced AI Engine:Uses a robust **Gradient Boosting Classifier** to analyze complex financial patterns.
*  Fair Assessment: Evaluates critical factors like Debt-to-Income (DTI) Ratio, FICO Score, and Employment History.
*  Interactive Dashboard: Features a modern "Dark Mode" UI with:
    * Credit Health Gauge: Visualizes the applicant's credit score zone.
    * DTI Thermometer:Instantly flags if a loan burden is too high for the income.
*  Decision Support: Provides clear "APPROVE" or "DECLINE"recommendations with calculated probability confidence.

# Tech Stack
* Core Logic:Python 3.10+
* Machine Learning: Scikit-Learn (Gradient Boosting, Random Forest)
* Data Processing: Pandas, NumPy
* Visualization: Plotly (Interactive financial charts)
* User Interface: Gradio (Web-based dashboard)
* Deployment:Render Cloud Hosting

# Project Structure
- app.py: The main application script containing the UI and business logic.
- credit_risk_model.pkl: The trained predictive model.
- credit_scaler.pkl: StandardScaler for normalizing input data.
-le_home.pkl & le_intent.pkl: Label encoders for categorical data.
- requirements.txt: List of dependencies for cloud deployment.

# Model Performance
The model was trained on a comprehensive financial dataset and optimized to handle imbalanced classes (default vs. non-default).

* Accuracy: ~92%
* Key Predictors:
    1.  Credit Score: Strongest indicator of past repayment behavior.
    2.  Income vs. Loan Amount: High debt burden significantly increases risk.
    3.  Previous Defaults: Historical behavior is a major risk factor.

## 🚀 How to Run Locally
If you want to run this on your own machine:

1.  Clone the repository:
    bash
    git clone (https://github.com/ashish-7561/credit-risk-dashboard.git)
    cd credit-risk-dashboard
   

2.  Install dependencies:
    bash
    pip install -r requirements.txt

3.  Run the App:
    bash
    python app.py
     the app will open in your browser at http://127.0.0.1:7860.

---
*Developed as part of the Machine Learning Capstone Project (Level 2).*
