# 🔧 Pump Failure Prediction with Anomaly Detection

This project was developed during my **Summer 2025 internship at Oil and Natural Gas Corporation Limited (ONGC)**.  
It implements a **Pump Failure Prediction System** using **machine learning anomaly detection techniques** to minimize downtime and improve predictive maintenance.

---

## 📌 Project Overview
- Worked with **historical pump operational data** (CSV sensor logs).  
- Preprocessed raw sensor data (cleaning, timestamp parsing, sensor classification).  
- Engineered basic time-based features (hour, day, weekday).  
- Applied **anomaly detection models**:
  - 🌲 **Isolation Forest**  
  - 🖥 **One-Class SVM**
- Combined predictions to identify likely pump failures.  
- Evaluated results with a simulated `true_failure` variable for demonstration.  

---

## ⚙️ Tech Stack
- **Languages:** Python (Pandas, NumPy)  
- **ML Libraries:** scikit-learn (Isolation Forest, One-Class SVM, StandardScaler, Classification Report)  
- **Other:** Matplotlib / Seaborn (optional for visualization)  

---

## 🛠 How It Works
1. **Data Ingestion**:  
   - Reads multiple `.csv` files from the `data/` folder.  
   - Expected columns: `name`, `date_time`, `value`, `machine_on`.  

2. **Preprocessing**:  
   - Parses timestamps, removes invalid rows.  
   - Classifies sensors into: `vibration`, `temperature`, `pressure`.  
   - Pivots into a wide table format.  

3. **Feature Engineering**:  
   - Adds temporal features: `hour`, `day`, `weekday`.  
   - Creates `true_failure` labels (simulated with 3% probability for testing).  

4. **Modeling**:  
   - **Isolation Forest** detects abnormal sensor behaviors.  
   - **One-Class SVM** provides alternative anomaly detection.  
   - Thresholds are set at the 5th percentile of anomaly scores.  
   - Final prediction = **union** of both models’ predictions.  

5. **Evaluation**:  
   - Prints a **classification report** (precision, recall, F1).  
   - Displays **Top 10 predicted failures** with sensor readings.  
   - Outputs the **next 100 pump predictions** for inspection.  
