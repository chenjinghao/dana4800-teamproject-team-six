import altair as alt
import streamlit as st
import pandas as pd
import numpy as np

st.title('DANA 4800 Team Project Phase 2')
st.subheader('Interactive Dashboard:computer:')

team_members = '''
- **Ananta Arora**
- **Jinghao Chen**
- **Roxanne Alvarez**
- **Teshani Jayasinghe**'''

tab1, tab2, tab3, tab4 = st.tabs(["Performance Comparision", "SHAP Analysis", "LIME Analysis", "Team Members"])

with tab1:
   pc_tab1, pc_tab2 = st.tabs(['Between models', 'Within models'])
   with pc_tab1:
    bm_num_sixseven,bm_num_twonine = st.tabs(["67 Features", "29 Features"])
    with bm_num_sixseven:
        model_cp_all_data = {
            'Models': ['SVM', 'Random Forest', 'XGBoost'],
            'Cross Val Score (mean)': [0.732416, 0.772202, 0.775205],
            'Std': [0.030066, 0.018957, 0.015806]}
        model_cp_all_df = pd.DataFrame(model_cp_all_data)
        st.dataframe(model_cp_all_df)
        st.image("image/pc/bw_sixseven.png")
        st.write("Classification Report")
        data = {
            'Metrics': ['Weighted Precision', 'Macro Precision', 'Weighted Recall', 'Macro Recall', 'Weighted F1 Score', 'Macro F1 Score', 'Accuracy', 'ROC AUC'],
            'SVC': [0.741185, 0.741185, 0.740769, 0.740769, 0.740657, 0.740657, None, None],
            'RandomForestClassifier': [0.777772, 0.777772, 0.777692, 0.777692, 0.777676, 0.777676, None, None],
            'XGBClassifier': [0.782603, 0.782603, 0.782308, 0.782308, 0.782251, None, None, None]}

        # Create the DataFrame
        df = pd.DataFrame(data)

        # Set the 'Metrics' column as the index
        df.set_index('Metrics', inplace=True)

        # Transpose the DataFrame
        df = df.T
        st.dataframe(df)
        st.write("ROC curve")
        st.image("image\pc\bw_sixseven_roc.png")

   with pc_tab2:
    wm_num_twonine, wm_num_sixseven = st.tabs(["67 Features", "29 Features"])

with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

with tab4:
    st.header("Team Members")
    st.markdown(team_members)











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



