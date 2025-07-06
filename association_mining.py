
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def show(df):
    st.header("Association Rule Mining")
    cols = st.multiselect("Select columns for Apriori", df.columns.tolist(), default=['inflight_wifi_service','online_boarding'])
    if not cols:
        st.warning("Select at least 2.")
        return
    data = df[cols].apply(lambda x: x.astype(str))
    ohe = pd.get_dummies(data)
    freq = apriori(ohe, min_support=0.1, use_colnames=True)
    conf = st.slider("Min confidence",0.1,1.0,0.6)
    rules = association_rules(freq, metric='confidence', min_threshold=conf)
    st.write(rules.head(10))
