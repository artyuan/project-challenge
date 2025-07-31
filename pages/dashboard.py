import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

logo_bottom = Image.open("images/phData.png")

df = pd.read_csv("data/prediction_logs_all_inputs.csv")

# ----- Streamlit App -----
st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")
st.title("🏡 House Price Prediction Monitoring")

# Define tabs
tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔸 Scatter Plot", "🔹 Box Plot"])

# Tab 1: Histogram
with tab1:
    st.subheader("Distribution of Predicted Prices")
    fig_hist = px.histogram(
        df,
        x="prediction",
        nbins=20,
        title="Distribution of Predicted Prices",
        labels={"predicted_price": "Predicted Price"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# Feature options
features = [col for col in df.columns if col not in ['id', 'timestamp', 'prediction','experiment_id','run_id']]

# Tab 2: Scatter Plot
with tab2:
    st.subheader("Scatter Plot: Feature vs Predicted Price")
    scatter_feature = st.selectbox("Select a feature for the scatter plot:", features, key="scatter")
    fig_scatter = px.scatter(
        df,
        x=scatter_feature,
        y="prediction",
        title=f"{scatter_feature} vs Predicted Price",
        labels={scatter_feature: scatter_feature, "predicted_price": "Predicted Price"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Tab 3: Box Plot
with tab3:
    st.subheader("Box Plot: Feature vs Predicted Price")
    box_feature = st.selectbox("Select a feature for the box plot:", features, key="box")
    fig_box = px.box(
        df,
        x=box_feature,
        y="prediction",
        title=f"Predicted Price by {box_feature}",
        labels={"predicted_price": "Predicted Price"}
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")
st.image(logo_bottom, width=200, caption="Powered by phData")