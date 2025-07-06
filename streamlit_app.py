
import streamlit as st
import pandas as pd
from modules import data_viz, classification, clustering, association_mining, regression

st.set_page_config(page_title="Ryanair Analytics Dashboard", layout="wide")

@st.cache_data
def load_data(path='airline_passenger_satisfaction.csv'):
    return pd.read_csv(path)

st.sidebar.title("Data Loader")
uploaded = st.sidebar.file_uploader("Upload CSV", type=['csv'])
data = load_data(uploaded) if uploaded else load_data()

st.title("Ryanair Analytics Dashboard")

tabs = st.tabs(["Data Visualization", "Classification", "Clustering", "Association Rules", "Regression"])

with tabs[0]:
    data_viz.show(data)

with tabs[1]:
    classification.show(data)

with tabs[2]:
    clustering.show(data)

with tabs[3]:
    association_mining.show(data)

with tabs[4]:
    regression.show(data)
