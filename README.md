
# 🌱 Sustainable Agriculture ML Project

## 📌 Overview

This project is a **Machine Learning-powered Crop Recommendation System**.
It predicts the most suitable crop for given **soil and climate conditions** (Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, rainfall).

The system consists of two parts:

* **Model Training (`train.py`)** → Trains & evaluates a Random Forest model with hyperparameter tuning, generates metrics & visualizations.
* **Streamlit App (`app.py`)** → Interactive UI where users can input soil/climate data and get crop recommendations with insights.



## ⚙️ Tech Stack

* **Python**
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib
* **Frontend:** Streamlit dashboard
* **ML Algorithm:** Random Forest Classifier with GridSearchCV



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
   * `metrics.json` (performance report)
   * `confusion_matrix.png`, `feature_importance.png`

4. **Run the Streamlit UI:**

   ```bash
   streamlit run app.py
   ```



## 📊 Features

✅ **Data Preprocessing & Scaling**
✅ **Random Forest with Hyperparameter Tuning (GridSearchCV)**
✅ **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, CV Score
✅ **Visualization:** Confusion Matrix & Feature Importance charts
✅ **Interactive Streamlit Dashboard** for crop recommendation
✅ **Top-3 Predicted Crops with Probabilities**
✅ **Sustainability Tips** for better farming practices

<img width="2000" height="1200" alt="crop_project_workflow" src="https://github.com/user-attachments/assets/e446a919-00a7-47ff-9eea-cadf3b34dbb9" />


## 📷 Demo

### 🔹 Dashboard Interface

<img width="1920" height="1020" alt="Screenshot 2025-09-06 024421" src="https://github.com/user-attachments/assets/a79e4d55-39bb-4f5c-9269-6c7b72f32850" />


### 🔹 Confusion Matrix

<img width="1920" height="1020" alt="Screenshot 2025-09-06 024350" src="https://github.com/user-attachments/assets/7f353301-8d04-4228-a3b9-29922a561d4a" />


### 🔹 Feature Importance

<img width="1920" height="1020" alt="Screenshot 2025-09-06 024404" src="https://github.com/user-attachments/assets/d9f878b1-0694-4158-9d81-2c8795629c80" />




👉 This project can be extended with **real-time weather APIs**, **farm advisory reports (PDF/CSV)**, and **deployment on cloud platforms** (Heroku, Render, AWS).


