
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def show(df):
    st.header("Quick Data Overview")
    st.write(df.head())
    st.write("Rows:", len(df))
    # Example plot
    fig, ax = plt.subplots()
    sns.countplot(x='satisfaction', data=df, ax=ax)
    st.pyplot(fig)
