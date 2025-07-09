import streamlit as st
import pandas as pd
import joblib
import shap
from clean_data import Clean_Data
import matplotlib.pyplot as plt

model = joblib.load("outputs/purchase_classifier_pipeline.joblib")
data = pd.read_parquet("data/processed/2019_Oct_clean.parquet")
clean = Clean_Data()
st.title("Customer Purchase Predictor")

data_X, data_y = clean.prepare_data(data)
probabilities = model.predict_proba(data_X)[:,1]


if st.checkbox("Show Raw User Data"):
    st.write(data.head())


with open("outputs/best_threshold.txt", "r") as file:
    best_threshold = float(file.readline().strip())

st.write(f"Threshold for buyer Prediction: **{best_threshold:.2f}**")
predictions = (probabilities >= best_threshold).astype(int)


def tag_lead(proba):
    if proba >= 0.8:
        return "🔥 Hot Lead"
    elif proba > 0.6:
        return "🌤️ Warm Lead"
    else:
        return "❄️ Cold Lead"
    
lead_scores = [tag_lead(p) for p in probabilities]

crm_table = pd.DataFrame({
    "User ID " : data["user_id"],
    "Purchase Probability" : probabilities,
    "Time of Activity" : data["hour_24"],
    "Predicted Buyer" : predictions.astype(bool),
    "Lead Score" : lead_scores
    
})

st.write("CRM Overview")
st.dataframe(crm_table)

csv = crm_table.to_csv(index=False)

buyer_rate_per_hour = crm_table.groupby("Time of Activity")["Predicted Buyer"].mean() * 100

st.line_chart(buyer_rate_per_hour, x_label="Time of Activity (24 hour time)", y_label="% of Predicted Buyers")
st.download_button("Download Lead Table", data=csv, file_name="crm_predictions.csv", mime="text/csv")

st.caption("Model trained on March 22 2025")
st.caption(f"Features used: {data_X.columns.tolist()}")