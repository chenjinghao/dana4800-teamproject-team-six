import altair as alt
import streamlit as st


"""
# DANA 4800 Team Project Phase 2
## Team 6
-	Ananta Arora (SID: 100421624)
-	Jinghao Chen (SID: 100406201)
-	Roxanne Alvarez (SID: 100405742)
-	Teshani Jayasinghe (SID: 100422405)
"""


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
