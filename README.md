# Customer Churn Prediction

### Objective

Predict which customers are likely to cancel their service and identify the main factors driving churn. The goal is to enable data-driven retention strategies that improve customer lifetime value and reduce revenue loss.

---

## 1. Business Context

Customer churn directly impacts profitability. Acquiring new users costs significantly more than retaining existing ones. A predictive model allows the company to identify at-risk customers early and take preventive actions such as special offers, loyalty programs, or proactive support.

---

## 2. Dataset

Source: [Kaggle – Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

**Key Fields**

* `customerID` – unique customer identifier
* `gender`, `SeniorCitizen`, `Partner`, `Dependents` – demographic info
* `tenure` – months with company
* `PhoneService`, `InternetService`, `StreamingTV`, etc. – service usage
* `Contract`, `PaperlessBilling`, `PaymentMethod` – contract and billing
* `MonthlyCharges`, `TotalCharges` – financial indicators
* `Churn` – target variable (Yes = churned, No = active)

---

## 3. Methodology

1. **Data Preprocessing**

   * Clean missing and inconsistent values
   * Encode categorical features
   * Scale numerical values
   * Split into train/test sets

2. **Exploratory Data Analysis**

   * Churn distribution by contract type, payment method, and tenure
   * Correlation analysis between service features and churn

3. **Modeling**

   * Baseline: Logistic Regression
   * Advanced: Random Forest, XGBoost, LightGBM
   * Evaluation using ROC-AUC, precision, recall, and F1-score

4. **Explainability**

   * Use SHAP values to identify top churn drivers
   * Visualize feature impact on churn probability

5. **Deployment (optional)**

   * Streamlit or FastAPI app for real-time churn prediction
   * Input: customer attributes
   * Output: churn probability and top influencing features

---

## 4. Repository Structure

```
customer-churn-prediction/
│
├── data/                # dataset or loading script
├── notebooks/           # exploratory and modeling notebooks
├── src/                 # preprocessing, training, and evaluation scripts
├── app/                 # Streamlit or FastAPI app
├── models/              # saved model files
├── requirements.txt
└── README.md
```

---

## 5. Evaluation Metrics

* **ROC-AUC** – measures model discrimination
* **F1-Score** – balances precision and recall
* **Recall** – key for identifying at-risk customers
* **Confusion Matrix** – visual summary of prediction accuracy

---

## 6. Example Results

| Model               | ROC-AUC | F1-Score | Recall |
| ------------------- | ------- | -------- | ------ |
| Logistic Regression | 0.82    | 0.63     | 0.72   |
| Random Forest       | 0.86    | 0.66     | 0.75   |
| XGBoost             | 0.88    | 0.68     | 0.77   |

**Top Churn Drivers (SHAP Analysis)**

* Short tenure
* Month-to-month contract
* High monthly charges
* Fiber optic internet
* Lack of tech support

---

## 7. How to Run

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
python src/train_model.py
streamlit run app/app.py
```

---

## 8. Future Work

* Add customer segmentation for targeted campaigns
* Integrate model predictions with CRM systems
* Extend to survival analysis for time-to-churn prediction
* Automate retraining with live data updates

---

Would you like me to create the **`requirements.txt`** and **`train_model.py` baseline script** next?
