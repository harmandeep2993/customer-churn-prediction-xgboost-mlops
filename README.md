## Customer Churn Prediction
An end-to-end machine learning project that predicts customer churn using XGBoost, built with a modular pipeline, Jupyter notebooks for analysis, and a Streamlit web app for deployment. This project demonstrates practical skills in data preprocessing, model tuning, evaluation, and deployment.

### Objective
Predict whether a customer is likely to churn (cancel their service) based on account and usage features. The goal is to help businesses reduce customer loss and improve retention through data-driven actions.

### Business Impact
- Customer retention costs are significantly lower than acquiring new users.  
- Early churn detection enables targeted offers and proactive customer support.  
- Predictive analytics improve customer lifetime value and overall profitability.

### Project Structure
```bash
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── Customer-Churn.csv             # Original dataset
│   └── processed/
│       └── churn_cleaned.csv              # Processed dataset
│
├── models/
│   ├── xgb_churn_full_tuned.pkl           # Final tuned model
│   ├── onehot_encoder.pkl                 # Saved encoder
│   └── train_columns.pkl                  # Training feature list
│
├── notebooks/
│   ├── eda_preprocess.ipynb               # EDA & preprocessing exploration
│   ├── train_model.ipynb                  # Model training & evaluation
│
├── src/
│   ├── preprocess.py                      # Data preprocessing pipeline
│   ├── train_model.py                     # Model training logic
│   ├── evaluate_model.py                  # Evaluation metrics
│   ├── predict.py                         # Inference pipeline
│   ├── pipeline.py                        # Full automation script
│
├── app.py                                 # Streamlit web app
├── requirements.txt                       # Dependencies
└── README.md                              # Project documentation
````
### Project Workflow

1. **Exploratory Data Analysis (EDA)**
   * Analyzed feature distributions and churn trends.
   * Identified missing values and data imbalances.

2. **Preprocessing Pipeline**
   * Encoded categorical and binary features.
   * Scaled numerical variables.
   * Built modular functions for reproducibility.

3. **Model Training and Tuning**
   * Compared baseline models (Random Forest, XGBoost).
   * Used RandomizedSearchCV for hyperparameter tuning.
   * Handled class imbalance with `scale_pos_weight`.

4. **Model Evaluation**
   * Measured performance with accuracy, precision, recall, F1-score, and confusion matrix.
   * Selected the tuned XGBoost model (Round 2) for final deployment.

5. **Deployment**
   * Developed a Streamlit web app for live predictions.
   * Users can input customer data and receive churn probability.

### Model Performance

| Metric            | Round 1 | Round 2 (Final) |
| :---------------- | :-----: | :-------------: |
| Accuracy          |   0.75  |       0.78      |
| Recall (Churn)    |   0.81  |       0.73      |
| Precision (Churn) |   0.52  |       0.56      |
| F1 (Churn)        |   0.64  |       0.63      |

**Final Model:** Tuned XGBoost (Round 2) — more balanced precision and recall.

### Key Learnings

* Designed a modular, reusable ML pipeline.
* Handled class imbalance using `scale_pos_weight`.
* Tuned hyperparameters effectively with cross-validation.
* Built and deployed a working Streamlit web app for predictions.
* Practiced clean project structure and documentation for reproducibility.

---

### Tech Stack

* Python 3.10+
* Pandas, NumPy, Scikit-learn, XGBoost
* Matplotlib, Seaborn
* Streamlit
* Joblib

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate         # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Training Pipeline

```bash
python -m src.pipeline
```

### 5. Launch the Streamlit App

```bash
streamlit run app.py
```

---

### Example App Output

```
Prediction Result:
No Churn
Churn Probability: 22.50 %
```

---

### Future Improvements

* Add explainability using SHAP or LIME.
* Automate retraining with live data updates.
* Integrate model insights with CRM systems.
* Containerize and deploy via Docker or AWS.

---

### Author

**Harman Singh**
Machine Learning & Data Science Enthusiast
Based in Germany
[LinkedIn](https://www.linkedin.com/in/) • [GitHub](https://github.com/)

```
