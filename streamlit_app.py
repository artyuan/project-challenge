import streamlit as st

pages={
    "House Pricing App": [
        st.Page('pages/price_prediction.py', title='House Price Prediction'),
        st.Page('pages/dashboard.py', title='Monitoring')
    ]
}

pg = st.navigation(pages)
pg.run()