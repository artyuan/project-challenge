import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
from PIL import Image
import pandas as pd
from src.logger import log_prediction


API_URL = "http://localhost:8000/predict"
API_URL_FULL = "http://localhost:8000/predict_full"
USERNAME = st.secrets["API_USERNAME"]
PASSWORD = st.secrets["API_PASSWORD"]

logo_top = Image.open("images/sound_realty.png")
logo_bottom = Image.open("images/phData.png")

col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image(logo_top, width=200)

with col_title:
    st.title("House Price Predictor")

st.divider()
st.subheader("Enter the property features:")

mode = st.radio("Choose input mode:", ["Manual Input", "Upload CSV"])

if mode == "Manual Input":
    st.subheader("Enter property features manually:")

    col1, col2 = st.columns(2)
    with col1:
        zipcode = st.number_input("Zipcode", step=1, value=98042)
        bedrooms = st.number_input("Bedrooms", min_value=0.0, value=4.0)
        sqft_living = st.number_input("Sqft Living", min_value=0.0, value=1680.0)
        floors = st.number_input("Floors", min_value=0.0, value=1.5)
    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=0.0, value=1.0)
        sqft_lot = st.number_input("Sqft Lot", min_value=0.0, value=5043.0)
        sqft_above = st.number_input("Sqft Above", min_value=0.0, value=1680.0)
        sqft_basement = st.number_input("Sqft Basement", min_value=0.0, value=1911.0)

    if st.button("Predict"):
        input_data = {
            "zipcode": zipcode,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement
        }

        try:
            response = requests.post(
                API_URL,
                json=input_data,
                auth=HTTPBasicAuth(USERNAME, PASSWORD)
            )

            if response.status_code == 200:
                result = response.json()
                log_prediction(input_data=result, full=False)
                st.success(f"💰 Predicted Price: ${result['prediction'][0]:,.2f}")
                st.caption(f"Request ID: {result['id']}")
            else:
                st.error(f"❌ Error: {response.status_code}")
                st.json(response.json())
        except Exception as e:
            st.error(f"Request failed: {e}")

else:
    st.subheader("Upload CSV for bulk predictions")
    st.markdown(
        "Expected columns: `zipcode`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `sqft_above`, `sqft_basement`")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Submit for Predictions"):
            df_results = pd.DataFrame()
            for idx, row in df.iterrows():
                input_data = row.to_dict()
                try:
                    response = requests.post(
                        API_URL_FULL,
                        json=input_data,
                        auth=HTTPBasicAuth(USERNAME, PASSWORD)
                    )
                    if response.status_code == 200:
                        result = response.json()
                        log_prediction(input_data=result, full=True)
                    else:
                        prediction = f"Error: {response.status_code}"
                except Exception as e:
                    prediction = f"Failed: {e}"
                predictions = pd.json_normalize(result)
                predictions = predictions.explode('prediction', ignore_index=True)
                df_results = pd.concat([df_results,predictions], axis=0).reset_index(drop=True)

            df = pd.concat([df_results, df], axis=1)
            st.success("✅ Predictions complete")
            st.dataframe(df)

            csv_download = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", data=csv_download, file_name="predictions_result.csv",
                               mime="text/csv")

st.markdown("---")
st.image(logo_bottom, width=200, caption="Powered by phData")