
import streamlit as st
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

def preprocess(df):
    df_num = df.select_dtypes(include=[np.number]).dropna(axis=1)
    target = st.selectbox("Select target numeric column", df_num.columns, index=0)
    X = df_num.drop(columns=[target])
    y = df_num[target]
    return train_test_split(X, y, test_size=0.2, random_state=42), target

def show(df):
    st.header("Regression Insights")
    (X_train, X_test, y_train, y_test), target = preprocess(df)
    models = {'Ridge': Ridge(), 'Lasso': Lasso(), 'DecisionTree': DecisionTreeRegressor()}
    results=[]
    for name,m in models.items():
        m.fit(X_train, y_train)
        pred=m.predict(X_test)
        results.append({'Model':name,'R2': r2_score(y_test,pred)})
    st.table(pd.DataFrame(results))
