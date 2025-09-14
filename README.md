
# 🌱 Sustainable Agriculture ML Project

## 📌 Overview

This project is a **Machine Learning-powered Crop Recommendation System** that helps farmers make sustainable decisions.
It predicts the most suitable crop based on **soil and climate conditions** (Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, rainfall).

The system consists of two main parts:

* **Model Training (`train.py`)** → Trains & evaluates a Random Forest model, scales features, saves encoders, and outputs performance metrics.
* **Streamlit App (`app.py`)** → Interactive dashboard that allows users to input soil/climate data, upload CSVs for batch prediction, and explore model insights (metrics, confusion matrix, feature importance).

---

## ⚙️ Tech Stack

* **Python**
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib
* **Frontend:** Streamlit interactive UI
* **ML Algorithm:** Random Forest Classifier

---

## 🚀 How to Run

1. **Clone this repo:**

   ```bash
   git clone https://github.com/your-username/crop-recommendation-ml.git
   cd crop-recommendation-ml
   ```

2. **Install requirements:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (optional, only if dataset updated):**

   ```bash
   python train.py
   ```

   This will generate:

   * `crop_model.pkl`, `scaler.pkl`, `label_encoder.pkl`

4. **Run the Streamlit App:**

   ```bash
   streamlit run app.py
   ```

---

## 📊 Features

✅ **Data Preprocessing & Scaling**
✅ **Random Forest Classification**
✅ **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score
✅ **Visualization:** Confusion Matrix & Feature Importance (in sidebar)
✅ **Interactive Streamlit Dashboard** for crop recommendation
✅ **Top-3 Predicted Crops with Probabilities**
✅ **Sustainability Tips** tailored to recommended crops
✅ **Batch Prediction** via CSV upload (with downloadable results)

---

## 🔄 Project Workflow

<img width="1536" height="1024" alt="workflow" src="https://github.com/user-attachments/assets/726e6385-9ca5-487a-9843-564d78eb7a5c" />

---

## 📷 Demo

### 🔹 Dashboard Interface

<img width="1920" height="1020" alt="Screenshot 2025-09-14 033234" src="https://github.com/user-attachments/assets/a5b0c15d-876c-4760-a121-1971beb7632e" />  
<img width="1920" height="1020" alt="Screenshot 2025-09-14 033329" src="https://github.com/user-attachments/assets/12f5344f-c47d-4093-83fd-22e26c4d2773" />  

### 🔹 Confusion Matrix

<img width="1920" height="1020" alt="Screenshot 2025-09-14 033249" src="https://github.com/user-attachments/assets/a57fa8f6-ebea-42bb-ae65-617c580d21f0" />  

### 🔹 Feature Importance

<img width="1920" height="1020" alt="Screenshot 2025-09-14 033307" src="https://github.com/user-attachments/assets/1aeae2e6-765e-46bf-9012-97670911f6b4" />  

---

👉 **Future Scope**

* Integration with **real-time weather APIs**
* **Farm advisory reports** (PDF/CSV)
* **Cloud deployment** (Heroku, Render, AWS, GCP)

