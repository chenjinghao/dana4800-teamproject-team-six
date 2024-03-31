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
        data = {"MLA used":"SVC","Weighted Precision":0.7411853826,"Macro Precision":0.7411853826,"Weighted Recall":0.7407692308,"Macro Recall":0.7407692308,"Weighted F1 Score":0.7406573605,"Macro F1 Score":0.7406573605,"Accuracy":0.7407692308,"ROC_AUC":0.8211704142},{"MLA used":"RandomForestClassifier","Weighted Precision":0.7855799588,"Macro Precision":0.7855799588,"Weighted Recall":0.7853846154,"Macro Recall":0.7853846154,"Weighted F1 Score":0.7853479086,"Macro F1 Score":0.7853479086,"Accuracy":0.7853846154,"ROC_AUC":0.8605136095},{"MLA used":"XGBClassifier","Weighted Precision":0.7826026693,"Macro Precision":0.7826026693,"Weighted Recall":0.7823076923,"Macro Recall":0.7823076923,"Weighted F1 Score":0.7822508714,"Macro F1 Score":0.7822508714,"Accuracy":0.7823076923,"ROC_AUC":0.8589798817}
        # Create the DataFrame
        df = pd.DataFrame(data)

        # Set the 'Metrics' column as the index
        df.set_index('MLA used', inplace=True)
        
        # Transpose the DataFrame
        df = df.transpose()
        st.dataframe(df)
        st.write("ROC curve")
        st.image("image/pc/bw_sixseven_roc.png")
    with bm_num_twonine:
        model_cp_twonine_data = {
            'Models': ['SVM', 'Random Forest', 'XGBoost'],
            'Cross Val Score (mean)': [0.732416, 0.772202, 0.775205],
            'Std': [0.030066, 0.018957, 0.015806]}
        model_cp_twonine_df = pd.DataFrame(model_cp_twonine_data)
        st.dataframe(model_cp_twonine_df)
        st.image("image/pc/bw_twonine.png")
        st.write("Classification Report")
        data = {"MLA used":"RandomForestClassifier","Train Accuracy":1.0,"Test Accuracy":0.7685,"Weighted Precision":0.7684774247,"Macro Precision":0.7684774247,"Weighted Recall":0.7684615385,"Macro Recall":0.7684615385,"Weighted F1 Score":0.7684581133,"Macro F1 Score":0.7684581133,"Accuracy":0.7684615385,"ROC_AUC":0.8479491124},{"MLA used":"XGBClassifier","Train Accuracy":1.0,"Test Accuracy":0.7562,"Weighted Precision":0.7562272272,"Macro Precision":0.7562272272,"Weighted Recall":0.7561538462,"Macro Recall":0.7561538462,"Weighted F1 Score":0.7561363861,"Macro F1 Score":0.7561363861,"Accuracy":0.7561538462,"ROC_AUC":0.832643787},{"MLA used":"SVC","Train Accuracy":0.7407,"Test Accuracy":0.75,"Weighted Precision":0.7507269598,"Macro Precision":0.7507269598,"Weighted Recall":0.75,"Macro Recall":0.75,"Weighted F1 Score":0.7498186555,"Macro F1 Score":0.7498186555,"Accuracy":0.75,"ROC_AUC":0.8258721893}
        # Create the DataFrame
        df = pd.DataFrame(data)

        # Set the 'Metrics' column as the index
        df.set_index('MLA used', inplace=True)
        
        # Transpose the DataFrame
        df = df.transpose()
        st.dataframe(df)
        st.write("ROC curve")
        st.image("image/pc/bw_twonine_roc.png")
   with pc_tab2:
    xgboost,randomforest,svc = st.tabs(["XGBoost", "RandomForest", "SVC"])
    with xgboost:
        model_cp_all_data = {
            'Models': ['SVM', 'Random Forest', 'XGBoost'],
            'Cross Val Score (mean)': [0.732416, 0.772202, 0.775205],
            'Std': [0.030066, 0.018957, 0.015806]}
        model_cp_all_df = pd.DataFrame(model_cp_all_data)
        st.dataframe(model_cp_all_df)
        st.image("image/pc/bw_sixseven.png")
        st.write("Classification Report")
        data = {"MLA used":"SVC","Weighted Precision":0.7411853826,"Macro Precision":0.7411853826,"Weighted Recall":0.7407692308,"Macro Recall":0.7407692308,"Weighted F1 Score":0.7406573605,"Macro F1 Score":0.7406573605,"Accuracy":0.7407692308,"ROC_AUC":0.8211704142},{"MLA used":"RandomForestClassifier","Weighted Precision":0.7855799588,"Macro Precision":0.7855799588,"Weighted Recall":0.7853846154,"Macro Recall":0.7853846154,"Weighted F1 Score":0.7853479086,"Macro F1 Score":0.7853479086,"Accuracy":0.7853846154,"ROC_AUC":0.8605136095},{"MLA used":"XGBClassifier","Weighted Precision":0.7826026693,"Macro Precision":0.7826026693,"Weighted Recall":0.7823076923,"Macro Recall":0.7823076923,"Weighted F1 Score":0.7822508714,"Macro F1 Score":0.7822508714,"Accuracy":0.7823076923,"ROC_AUC":0.8589798817}
        # Create the DataFrame
        df = pd.DataFrame(data)

        # Set the 'Metrics' column as the index
        df.set_index('MLA used', inplace=True)
        
        # Transpose the DataFrame
        df = df.transpose()
        st.dataframe(df)
        st.write("ROC curve")
        st.image("image/pc/bw_sixseven_roc.png")
    with randomforest:
        model_cp_twonine_data = {
            'Models': ['SVM', 'Random Forest', 'XGBoost'],
            'Cross Val Score (mean)': [0.732416, 0.772202, 0.775205],
            'Std': [0.030066, 0.018957, 0.015806]}
        model_cp_twonine_df = pd.DataFrame(model_cp_twonine_data)
        st.dataframe(model_cp_twonine_df)
        st.image("image/pc/bw_twonine.png")
        st.write("Classification Report")
        data = {"MLA used":"RandomForestClassifier","Train Accuracy":1.0,"Test Accuracy":0.7685,"Weighted Precision":0.7684774247,"Macro Precision":0.7684774247,"Weighted Recall":0.7684615385,"Macro Recall":0.7684615385,"Weighted F1 Score":0.7684581133,"Macro F1 Score":0.7684581133,"Accuracy":0.7684615385,"ROC_AUC":0.8479491124},{"MLA used":"XGBClassifier","Train Accuracy":1.0,"Test Accuracy":0.7562,"Weighted Precision":0.7562272272,"Macro Precision":0.7562272272,"Weighted Recall":0.7561538462,"Macro Recall":0.7561538462,"Weighted F1 Score":0.7561363861,"Macro F1 Score":0.7561363861,"Accuracy":0.7561538462,"ROC_AUC":0.832643787},{"MLA used":"SVC","Train Accuracy":0.7407,"Test Accuracy":0.75,"Weighted Precision":0.7507269598,"Macro Precision":0.7507269598,"Weighted Recall":0.75,"Macro Recall":0.75,"Weighted F1 Score":0.7498186555,"Macro F1 Score":0.7498186555,"Accuracy":0.75,"ROC_AUC":0.8258721893}
        # Create the DataFrame
        df = pd.DataFrame(data)

        # Set the 'Metrics' column as the index
        df.set_index('MLA used', inplace=True)
        
        # Transpose the DataFrame
        df = df.transpose()
        st.dataframe(df)
        st.write("ROC curve")
        st.image("image/pc/bw_twonine_roc.png")
    with svc:
        st.write('updating')
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



