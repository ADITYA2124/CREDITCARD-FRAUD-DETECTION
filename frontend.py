# frontend.py
import streamlit as st
import requests
import pandas as pd
import io
import base64

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")

# ==================== Sidebar ====================
st.sidebar.title("Options")
mode = st.sidebar.radio("Choose Input Mode:", ["Single Transaction (JSON)", "Bulk CSV Upload"])
model_choice = st.sidebar.selectbox("Choose Model:", ["rf", "xgb"])

# ==================== Single Transaction ====================
if mode == "Single Transaction (JSON)":
    st.subheader("Enter Transaction (JSON format)")
    
    # Default sample JSON
    default_json = """{
      "transactions": [
        {
          "Time": 406,
          "V1": -2.3122265423263,
          "V2": 1.95199201064158,
          "V3": -1.60985073229769,
          "V4": 3.9979055875468,
          "V5": -0.522187864667764,
          "V6": -1.42654531920595,
          "V7": -2.53738730624579,
          "V8": 1.39165724829804,
          "V9": -2.77008927782766,
          "V10": -2.77227214465915,
          "V11": 3.20203320709635,
          "V12": -2.89990738849473,
          "V13": -0.595221881324605,
          "V14": -4.28925378244217,
          "V15": 0.389724120274487,
          "V16": -1.14074717980657,
          "V17": -2.83005567450437,
          "V18": -0.0168224681808257,
          "V19": 0.416955705037907,
          "V20": 0.126910559061474,
          "V21": -0.422911082702381,
          "V22": 0.341928035575878,
          "V23": 0.253414715956928,
          "V24": -0.129229121167160,
          "V25": 0.390146124030295,
          "V26": -0.140405148480124,
          "V27": -0.187764078267086,
          "V28": -0.00137011044780762,
          "Amount": 0.0
        }
      ]
    }"""
    
    user_json = st.text_area("Paste Transaction JSON:", default_json, height=300)

    if st.button("Predict"):
        try:
            response = requests.post(f"{API_URL}/predict?model={model_choice}", data=user_json, headers={"Content-Type": "application/json"})
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Model Used: {result['model_used']}")
                st.write(f"**Classification:** {result['classification']}")
                st.write(f"**Probability (Fraud):** {result['probability']:.4f}")

                # Show Graph
                graph_b64 = result["graph"]
                st.image(base64.b64decode(graph_b64), caption="Probability Distribution")
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ==================== Bulk CSV ====================
elif mode == "Bulk CSV Upload":
    st.subheader("Upload CSV with Transactions")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Processing CSV..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{API_URL}/predict_csv?model={model_choice}", files=files)

        if response.status_code == 200:
            result = response.json()

            # Processed CSV
            processed_csv = pd.read_csv(io.StringIO(result["processed_csv"]))

            st.success(f"✅ Processed {len(processed_csv)} transactions with {result['model_used']}")
            st.subheader("📊 Processed Transactions (Scrollable)")
            st.dataframe(processed_csv, height=500, use_container_width=True)

            # Fraud distribution graph
            st.subheader("Fraud vs Legit Predictions")
            st.image(base64.b64decode(result["graph"]))
        else:
            st.error(f"API Error: {response.text}")
