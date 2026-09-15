# 💾 Hard Drive Failure Prediction

A machine learning system that predicts hard drive failures up to 7 days in advance using real-world sensor data from the Backblaze Data Center.

The goal is to identify drives at risk of failure early enough to enable proactive replacement, reducing unexpected downtime, operational costs, and potential data loss.

---

## 📌 Problem Statement

Unexpected hard drive failures can lead to:

- Data loss and service disruption
- Unplanned hardware replacement
- Increased maintenance costs
- Reduced data-center reliability

This project uses historical hard drive health and SMART sensor data to predict whether a drive is likely to fail within the next 7 days.

---

## 📊 Dataset

Source: Backblaze Quarterly Hard Drive Stats
Period: January – June 2020
Scale: ~24M+ daily drive readings
Sensors: 6 selected SMART health indicators

The dataset contains daily observations of individual drives, including sensor readings, drive information, and failure events.

«🔗 Dataset: "Backblaze Hard Drive Data" (https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)»

---

## 🔬 Approach

### 1. Exploratory Data Analysis

Analyzed failure patterns across:

- Drive age
- Drive models
- Temperature
- SMART health indicators
- Failure frequency

### 2. Feature Engineering

Created features specifically designed for early failure detection:

- 7-day failure label — whether a drive fails within the following 7 days
- 7-day rolling features for key SMART sensors
- Sensor trends and historical behavior
- Time-based features

### 3. Model Development

Compared multiple approaches:

Model| Purpose
Logistic Regression| Baseline classification model
XGBoost| Main failure prediction model
Cox Survival Analysis| Time-to-failure analysis

### 4. Business-Aware Evaluation

Instead of selecting a threshold solely based on model metrics, a cost-based threshold framework was developed.

The final threshold selection was constrained by a minimum 60% recall requirement, ensuring that the model remains practical for proactive maintenance.

### 5. Interactive Dashboard

A Streamlit dashboard was developed to allow users to:

- Explore drive-level risk
- Filter drive information
- View failure predictions
- Examine model metrics
- Understand the business impact of predictions

---

## 🔍 Key Findings

### 🌡️ Temperature and Failure Risk

Failure rates increased significantly at higher temperatures, with failure rates rising approximately 5× above 40°C.

### 💽 Drive Model Reliability

Failure rates varied considerably between drive models. One consumer-grade model experienced approximately 24× the failure rate of enterprise drives in the analyzed dataset.

### 💰 Cost vs. Recall

Pure cost minimization selected an impractical threshold that resulted in only 11% recall.

Introducing a minimum 60% recall constraint produced a much more deployable solution by prioritizing the detection of actual failures while controlling false positives.

---

## 📈 Model Results

| Model | ROC-AUC | Recall |
|---|---:|---:|
| Logistic Regression (Baseline) | 0.57 | 0% |
| XGBoost | 0.79 | 54% |
| XGBoost (Tuned + Extra Features) | **0.81** | **57–60%** |

The tuned XGBoost model provided the strongest overall discrimination between drives that fail and drives that remain healthy.

---

## 💵 Business Impact

At the 60% minimum recall requirement, the final model achieved:

- 13% reduction in false positives
- Approximately $7.2M lower estimated cost
- Maintained approximately 60% recall

This demonstrates why model deployment should consider business costs and operational requirements, rather than optimizing a single machine-learning metric.

---

## 🖥️ Dashboard

The project includes an interactive Streamlit dashboard for exploring drive failure risk and model performance.

Run Locally

streamlit run app/dashboard.py

---

## 🛠️ Tech Stack

Programming & Data

- Python
- pandas
- NumPy

Machine Learning

- XGBoost
- scikit-learn
- lifelines

Visualization & Dashboard

- Plotly
- Streamlit

---
## 📂 Project Structure

```text
Hard-Drive-Failure-Prediction/
│
├── dashboard.py
│
├── disc prediction project.ipynb
│
├── dashboard_data.csv
├── dashboard_metrics.json
├── cost_curve_data.csv
│
├── requirements.txt
└── README.md
```

## 🚀 Future Improvements

1. Longer Rolling Windows

Extend the feature engineering pipeline from 7-day windows to 30-day rolling features to capture longer-term degradation patterns.

2. Larger Training Dataset

Train using a full year of Backblaze data to increase the number of failure examples and improve model generalization.

3. Automated Prediction Pipeline

Deploy the model as a scheduled batch-scoring pipeline that:

New Drive Data
      ↓
Feature Engineering
      ↓
Failure Risk Prediction
      ↓
Risk Threshold
      ↓
Flag At-Risk Drives
      ↓
Proactive Replacement

4. Production Deployment

Integrate the prediction system with data-center monitoring infrastructure to automatically score drives and generate maintenance alerts.

---

## 🎯 Project Takeaway

This project demonstrates an end-to-end approach to predictive maintenance, combining:

Real-world data → Feature Engineering → Machine Learning → Cost Optimization → Interactive Dashboard

Rather than treating hard drive failure prediction as only a classification problem, the project incorporates time-based prediction, business costs, and operational constraints to create a more practical predictive-maintenance solution.

---

## 👩‍💻 Author

### Nuha Mushtaq, Tabassum Fathima, Yerra Shailaja

Built as a machine learning and predictive-maintenance project using real Backblaze hard drive data.
