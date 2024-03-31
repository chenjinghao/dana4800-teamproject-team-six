import altair as alt
import streamlit as st

st.title('DANA 4800 Team Project Phase 2')

team_members = '''<ul>
<li>Yash Shah</li>
<li>Shivam Patel</li>'''

# Using "with" notation
with st.sidebar:
    st.title("Section")
    st.header("Performance Comparision")
    st.header("SHAP Analysis")
    st.header("LIME Analysis")
    st.header("Team Members")
    st.markdown(team_members)



