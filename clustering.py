
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def preprocess(df):
    num = df.select_dtypes(include=[np.number])
    return StandardScaler().fit_transform(num)

def show(df):
    st.header("K-means Clustering")
    data = preprocess(df)
    k = st.slider("Select number of clusters",2,10,3)
    km = KMeans(n_clusters=k, random_state=42).fit(data)
    st.write("Inertia:", km.inertia_)
    # elbows
    if st.checkbox("Show elbow plot"):
        inertias=[]
        ks=range(2,11)
        for i in ks:
            inertias.append(KMeans(n_clusters=i, random_state=42).fit(data).inertia_)
        fig, ax=plt.subplots()
        sns.lineplot(x=ks, y=inertias, ax=ax, marker='o')
        st.pyplot(fig)
