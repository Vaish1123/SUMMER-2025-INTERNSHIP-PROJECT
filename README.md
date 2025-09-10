# SUMMER-2025-INTERNSHIP-PROJECT
# 🔧 Pump Failure Prediction using Machine Learning

This project was developed during my **Summer 2025 internship at Oil and Natural Gas Corporation Limited (ONGC)**.  
The goal was to build a **Pump Failure Prediction System** to enhance maintenance efficiency, minimize unplanned downtime, and optimize resource allocation in large-scale industrial operations.

---

## 📌 Project Overview
- Designed and implemented a **predictive maintenance pipeline** for pump systems.  
- Processed and cleaned large **time-series sensor datasets** (vibration, temperature, pressure).  
- Engineered features (rolling averages, statistical metrics, interaction ratios).  
- Built anomaly detection models:
  - **Isolation Forest** 🌲
  - **One-Class SVM** 🖥
- Applied **Principal Component Analysis (PCA)** for dimensionality reduction & visualization.  
- Delivered **failure predictions + root cause indicators** for proactive maintenance scheduling.  

---

## ⚙️ Tech Stack
- **Languages:** Python (Pandas, NumPy, Matplotlib, Seaborn)  
- **ML Libraries:** Scikit-learn (Isolation Forest, One-Class SVM, PCA), Statsmodels  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment-ready design:** Modular pipeline with `load → preprocess → feature engineer → train → evaluate → export results`.

---

## 🚀 Key Results
- Achieved reliable anomaly detection on pump datasets.  
- Identified **sensor parameters most correlated with failures**.  
- Generated **confusion matrices, PCA plots, and feature importance charts**.  
- Final system predicted both **time-to-failure** and **likely causes** of failures.  

### Example Outputs:
- Confusion Matrix:  
  ![Confusion Matrix](results/confusion_matrix.png)

- PCA Visualization:  
  ![PCA](results/pca_visualization.png)

- Feature Importance:  
  ![Feature Importance](results/feature_importance.png)

---

## 📦 Setup & Usage
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/pump-failure-prediction.git
   cd pump-failure-prediction
