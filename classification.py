
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def preprocess(df):
    df = df.copy()
    df['target'] = df['satisfaction'].map({'satisfied':1,'neutral or dissatisfied':0})
    X = df.select_dtypes(include=[np.number]).drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_models():
    return {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'GBRT': GradientBoostingClassifier()
    }

def show(df):
    st.header("Classification Models")
    X_train, X_test, y_train, y_test = preprocess(df)
    results = []
    models = get_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        results.append({
            'Model': name,
            'Test Accuracy': report['accuracy'],
            'Precision': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1-score': report['1']['f1-score']
        })
    st.dataframe(pd.DataFrame(results))
