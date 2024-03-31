import altair as alt
import streamlit as st
import pandas as pd
import numpy as np

team_members = '''
- **Ananta Arora**
- **Jinghao Chen**
- **Roxanne Alvarez**
- **Teshani Jayasinghe**'''

tab1, tab2, tab3, tab4 = st.tabs(["Performance Comparision", "SHAP Analysis", "LIME Analysis", "Team Members"])

tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

with tab1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

with tab4:
    st.header("Team Members")
    st.markdown(team_members)



# model_cp_all_data = {
#     'Models': ['SVM', 'Random Forest', 'XGBoost'],
#     'Cross Val Score (mean)': [0.732416, 0.772202, 0.775205],
#     'Std': [0.030066, 0.018957, 0.015806]
# }
# model_cp_all_df = pd.DataFrame(model_cp_all_data)


# st.title('DANA 4800 Team Project Phase 2')
# st.subheader('Interactive Dashboard:computer:')
# st.dataframe(model_cp_all_df)



# # Using "with" notation
# with st.sidebar:
#     st.title("Section")

#     st.header("Performance Comparision")
#     st.subheader('Between Models')
#     num_features_option = st.selectbox(
#         "Number of features", 
#         ("67","29"), placeholder="Pick")
#     st.write(f'You selected: {num_features_option} features.' )
#     st.header("SHAP Analysis")
#     st.header("LIME Analysis")
#     st.header("Team Members")
#     st.markdown(team_members)



