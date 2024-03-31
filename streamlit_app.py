import altair as alt
import streamlit as st

# Loading Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from scipy import stats
import scipy.cluster.hierarchy


import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn import svm, model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_score, recall_score, auc,roc_curve, roc_auc_score, f1_score, accuracy_score
from imblearn.under_sampling import RandomUnderSampler

"""
# DANA 4800 Team Project Phase 2
"""


# Using "with" notation
with st.sidebar:
    st.title("Section")
    st.header("Performance Comparision")
    st.header("SHAP Analysis")
    st.header("LIME Analysis")
    st.header("Team Members")
    st.markdown("Ananta Arora (SID: 100421624)\
                Jinghao Chen (SID: 100406201)\
                Roxanne Alvarez (SID: 100405742)\
                Teshani Jayasinghe (SID: 100422405)")



