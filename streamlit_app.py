import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# DANA 4800 Team Project Phase 2
## Team 6
"""
st.sidebar.['Selection']

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )
