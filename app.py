import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

# Load the final saved model (update filename if needed)
model = joblib.load('final_xgboost_model.joblib')

# Load the model and feature names
model = joblib.load("final_xgboost_model.joblib")   # or gradient_boosting_model.joblib
feature_names = joblib.load("model_features.joblib")


# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_data(batch_df, apply_clusters=False):
    # Relevant columns
    relevant_columns = ['tenure', 'OnlineSecurity', 'OnlineBackup', 'TechSupport',
                        'Contract', 'MonthlyCharges', 'TotalCharges']
    batch_df = batch_df[relevant_columns]

    # Map categorical values
    batch_df['OnlineSecurity'] = batch_df['OnlineSecurity'].map({'No': 0, 'Yes': 2, 'No internet service': 1})
    batch_df['OnlineBackup'] = batch_df['OnlineBackup'].map({'No': 0, 'Yes': 2, 'No internet service': 1})
    batch_df['TechSupport'] = batch_df['TechSupport'].map({'No': 0, 'Yes': 2, 'No internet service': 1})
    batch_df['Contract'] = batch_df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

    # Ensure numeric
    batch_df = batch_df.apply(pd.to_numeric, errors='coerce')
    batch_df.fillna(0, inplace=True)

    # Only apply clustering if batch mode
    if apply_clusters and len(batch_df) > 5:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(batch_df)
        batch_df['KMeansCluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(scaled)
        batch_df['AggCluster'] = AgglomerativeClustering(n_clusters=5).fit_predict(scaled)

    # Align input with training features
    for col in feature_names:
        if col not in batch_df.columns:
            batch_df[col] = 0  # add missing columns with default value
    batch_df = batch_df[feature_names]  # reorder to match training
    return batch_df


# -----------------------------
# Prediction Function
# -----------------------------
def predict(model, input_df):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]  # Probability of churn
    return prediction[0], probability[0]

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Customer Segmentation & Churn Prediction", page_icon=":bar_chart:", layout="wide")
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox("Choose Mode:", ("Online", "Batch"))

    st.sidebar.info("This app predicts customer churn and shows segmentation insights.")

    if mode == "Online":
        st.title("Online Prediction (Single Customer)")
        st.subheader("Input Values")
        st.write("[0 = No, 1 = No Internet Service, 2 = Yes]")

        tenure = st.number_input('Tenure (Months):', min_value=1, max_value=72, value=1)
        OnlineSecurity = st.selectbox('Online Security:', ['No', 'Yes', 'No internet service'])
        OnlineBackup = st.selectbox('Online Backup:', ['No', 'Yes', 'No internet service'])
        TechSupport = st.selectbox('Tech Support:', ['No', 'Yes', 'No internet service'])
        Contract = st.selectbox('Contract Type:', ['Month-to-month', 'One year', 'Two year'])
        MonthlyCharges = st.number_input('Monthly Charges ($):', min_value=18, max_value=120, value=18)
        TotalCharges = st.number_input('Total Charges ($):', min_value=18, max_value=9000, value=18)

        input_dict = {
            'tenure': [tenure],
            'OnlineSecurity': [OnlineSecurity],
            'OnlineBackup': [OnlineBackup],
            'TechSupport': [TechSupport],
            'Contract': [Contract],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges]
        }

        input_df = pd.DataFrame(input_dict)
        processed_df = preprocess_data(input_df, apply_clusters=False)
        pred, prob = predict(model, processed_df)

        st.write("### Prediction Result")
        st.write("Churn" if pred == 1 else "Retention")
        st.write(f"Probability of Churn: {prob:.2f}")

    else:
        st.title("Batch Prediction (Upload CSV)")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            processed_df = preprocess_data(batch_df, apply_clusters=True)
            predictions = model.predict(processed_df)
            probabilities = model.predict_proba(processed_df)[:, 1]

            batch_df['ChurnPrediction'] = predictions
            batch_df['ChurnProbability'] = probabilities

            st.write("### Results")
            st.dataframe(batch_df.head())

            # Download results
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "churn_predictions.csv", "text/csv")

if __name__ == "__main__":
    main()
