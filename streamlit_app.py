import altair as alt
import streamlit as st
import pandas as pd

st.title('DANA 4800 Team Project Phase 2')
st.subheader('Interactive Dashboard:computer:')


team_members = '''
- **Ananta Arora**
- **Jinghao Chen**
- **Roxanne Alvarez**
- **Teshani Jayasinghe**'''



# Using "with" notation
with st.sidebar:
    st.title("Section")
    st.header("Performance Comparision")
    st.subheader('Between Models')
    num_features_option = st.selecbox('Select the number of features', [67,29])
    st.write('You selected:', num_features_option)
    st.header("SHAP Analysis")
    st.header("LIME Analysis")
    st.header("Team Members")
    st.markdown(team_members)



