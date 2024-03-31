import altair as alt
import streamlit as st
import pandas as pd

model_cp_all_data = {
    'Models': ['SVM', 'Random Forest', 'XGBoost'],
    'Cross Val Score (mean)': [0.732416, 0.772202, 0.775205],
    'Std': [0.030066, 0.018957, 0.015806]
}
model_cp_all_df = pd.DataFrame(model_cp_all_data)


st.title('DANA 4800 Team Project Phase 2')
st.subheader('Interactive Dashboard:computer:')
st.dataframe(model_cp_all_df)

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
    num_features_option = st.selectbox(
        "Select the number of features", 
        ("67","29"),)
    st.write(f'You selected: {num_features_option} features.' )
    st.header("SHAP Analysis")
    st.header("LIME Analysis")
    st.header("Team Members")
    st.markdown(team_members)



