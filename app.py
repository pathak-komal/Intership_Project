# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# --------------------------
# Load Trained Model + Preprocessors
# --------------------------
rf = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# --------------------------
# Reload dataset for evaluation
# --------------------------
df = pd.read_csv("Crop_recommendation.csv")
df["label_encoded"] = le.transform(df["label"])
X = df.drop(["label", "label_encoded"], axis=1)
y = df["label_encoded"]

# split like training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test_scaled = scaler.transform(X_test)

y_pred = rf.predict(X_test_scaled)

# compute metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
rec = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# --------------------------
# Streamlit Config
# --------------------------
st.set_page_config(page_title="Sustainable Crop Recommendation", page_icon="🌱", layout="wide")

# --------------------------
# Sidebar Section
# --------------------------
st.sidebar.title("📊 Model Insights")

with st.sidebar.expander("Model Performance", expanded=True):
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**Precision:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")

with st.sidebar.expander("Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, cmap="Blues", annot=False, xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with st.sidebar.expander("Feature Importance"):
    if hasattr(rf, "feature_importances_"):
        importances = rf.feature_importances_
        features = ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"]
        imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(
            by="Importance", ascending=False
        )
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax2, palette="viridis")
        st.pyplot(fig2)

# --------------------------
# Main Section with Tabs
# --------------------------
tab1, tab2 = st.tabs(["🌾 Crop Prediction", "ℹ️ About Project"])

with tab1:
    st.title("🌱 Sustainable Agriculture Assistant")
    st.write("Enter your soil & climate details to get the best **crop recommendation**")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen content (N)", min_value=0, max_value=200, value=50)
        P = st.number_input("Phosphorus content (P)", min_value=0, max_value=200, value=50)
        K = st.number_input("Potassium content (K)", min_value=0, max_value=200, value=50)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)

    with col2:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

    if st.button("🌾 Recommend Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        probs = rf.predict_proba(input_scaled)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_crops = le.inverse_transform(top3_idx)
        top3_probs = probs[top3_idx] * 100

        best_crop = top3_crops[0]
        st.success(f"✅ Recommended Sustainable Crop: **{best_crop.capitalize()}**")

        st.write("### 📊 Top 3 Crop Probabilities")
        prob_df = pd.DataFrame({"Crop": top3_crops, "Probability (%)": top3_probs})
        st.bar_chart(prob_df.set_index("Crop"))

        with st.expander("🌍 Sustainability Tips"):
            if best_crop in ["rice", "sugarcane"]:
                st.info("💧 These crops need lots of water. Ensure efficient irrigation!")
            elif best_crop in ["maize", "millet", "barley"]:
                st.info("🌾 These crops are climate-resilient and eco-friendly.")
            else:
                st.info("🌍 Rotate this crop with legumes for soil health improvement.")

    st.markdown("---")

    st.write("### 📂 Batch Prediction (Upload CSV)")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with columns: N, P, K, temperature, humidity, ph, rainfall",
        type="csv",
    )
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        df_scaled = scaler.transform(df_upload.values)
        preds = rf.predict(df_scaled)
        df_upload["Predicted Crop"] = le.inverse_transform(preds)
        st.write("✅ Batch Prediction Results")
        st.dataframe(df_upload)

        csv = df_upload.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Results as CSV",
            data=csv,
            file_name="crop_predictions.csv",
            mime="text/csv",
        )

with tab2:
    st.header("ℹ️ About Project")
    st.markdown(
        """
    This project demonstrates how **Machine Learning** can assist farmers in making 
    sustainable crop choices.  

    **Key Features**  
    - Crop recommendation using soil & climate data  
    - Model insights (metrics, confusion matrix, feature importance)  
    - Batch prediction via CSV upload  
    - Sustainability tips for crops  

    **Tech Stack**  
    - Python, scikit-learn  
    - Streamlit (UI)  
    - Matplotlib, Seaborn, Pandas, NumPy  
    """
    )
