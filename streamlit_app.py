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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Performance Comparision", "SHAP Analysis", "LIME Analysis", "Permutation Feature Analysis","Team Members"])

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
        model_cp_xgb_data = {
            'Models': ['All Features', '29 Features'],
            'Cross Val Score (mean)': [0.777210, 0.786471],
            'Std': [0.026401, 0.023387]
}
        model_cp_xgb_df = pd.DataFrame(model_cp_xgb_data)
        st.dataframe(model_cp_xgb_df)
        st.image("image/pc/wm_xgboost.png")
        st.write("Classification Report")
        data = {
            'Feature': [
                'Weighted Precision', 'Macro Precision', 'Weighted Recall', 
                'Macro Recall', 'Weighted F1 Score', 'Macro F1 Score', 
                'Accuracy', 'ROC_AUC'
            ],
            'All Features': [
                0.782602669, 0.782602669, 0.782307692, 
                0.782307692, 0.782250871, 0.782250871, 
                0.782307692, 0.858979882
            ],
            '29 Features': [
                0.756227227, 0.756227227, 0.756153846, 
                0.756153846, 0.756136386, 0.756136386, 
                0.756153846, 0.832643787
            ]}        
        # Create the DataFrame
        df = pd.DataFrame(data)

        # Set the 'Metrics' column as the index
        df.set_index('Feature', inplace=True)
        
        # Transpose the DataFrame
        st.dataframe(df)
        st.write("ROC curve")
        st.image("image/pc/wm_xgboost_roc.png")
    with randomforest:
        model_cp_rf_data = {
            'Models': ['All Features', '29 Features'],
            'Cross Val Score (mean)': [0.775211, 0.775865],
            'Std': [0.019693, 0.017377]}
        model_cp_rf_df = pd.DataFrame(model_cp_rf_data)
        st.dataframe(model_cp_rf_df)
        st.image("image/pc/wm_rf.png")
        st.write("Classification Report")
        data = {
            'Feature': ['Weighted Precision', 'Macro Precision', 'Weighted Recall', 
                        'Macro Recall', 'Weighted F1 Score', 'Macro F1 Score',
                        'Accuracy', 'ROC_AUC'],
            'All Features': [0.776965031, 0.776965031, 0.776923077, 
                            0.776923077, 0.776914629, 0.776914629,
                            0.776923077, 0.858069822],
            '29 Features': [0.769271558, 0.769271558, 0.769230769,
                            0.769230769, 0.76922203, 0.76922203,
                            0.769230769, 0.849074556]
        }        # Create the DataFrame
        df = pd.DataFrame(data)

        # Set the 'Metrics' column as the index
        df.set_index('Feature', inplace=True)
        
        # Transpose the DataFrame
        st.dataframe(df)
        st.write("ROC curve")
        st.image("image/pc/wm_rf_roc.png")
    with svc:
        model_cp_svc_data = {
            'Models': ['All Features', '29 Features'],
            'Cross Val Score (mean)': [0.731407, 0.731099],
            'Std': [0.023415, 0.022104]}
        model_cp_svc_df = pd.DataFrame(model_cp_svc_data)
        st.dataframe(model_cp_rf_df)
        st.image("image/pc/wm_svc.png")
        st.write("Classification Report")
        data = {
            'Feature': ['Weighted Precision', 'Macro Precision', 'Weighted Recall', 
                        'Macro Recall', 'Weighted F1 Score', 'Macro F1 Score',
                        'Accuracy', 'ROC_AUC'],
            'All Features': [0.741185383, 0.741185383, 0.740769231, 
                            0.740769231, 0.74065736, 0.74065736,
                            0.740769231, 0.821170414],
            '29 Features': [0.75072696, 0.75072696, 0.75, 0.75,
                            0.749818656, 0.749818656, 0.75 , 0.825887574]}
        df = pd.DataFrame(data)

        # Set the 'Metrics' column as the index
        df.set_index('Feature', inplace=True)
        
        # Transpose the DataFrame
        st.dataframe(df)
        st.write("ROC curve")
        st.image("image/pc/wm_svc_roc.png")
with tab2:
   ft_imp, ft_sixseven, ft_twonine = st.tabs(["Feature Important", "67 Features", "29 Features"])
   with ft_imp:
        col1, col2 = st.columns(2)
        with col1:
            st.header("67 Features")
            st.image("image/pc/ft_67.png")
        with col2:
            st.header("29 Features")
            st.image("image/pc/ft_29.png")
   with ft_sixseven:
        tp, fp, tn, fn, heatm, bees, f_cluster = st.tabs([
            "True Positive", 
            "False Positive", 
            "True Negative", 
            "False Negative", 
            "Heat Map Matrix", 
            "Bee Swarm Plot",
            "Feature Clustering"
            ])
        with tp:
            st.image("shap67tp.png")
        with fp:
            st.image("shap67fp.png")
        with tn:
            st.image("shap67tn.png")
        with fn:
            st.image("shap67fn.png")
        with heatm:
            st.image("shap67heatm.png")
        with bees:
            st.image("shap67bees.png")
        with f_cluster:
            st.image("shap67f_clustering.png")
   with ft_twonine:
        tp, fp, tn, fn, heatm, bees, f_cluster = st.tabs([
            "True Positive", 
            "False Positive", 
            "True Negative", 
            "False Negative", 
            "Heat Map Matrix", 
            "Bee Swarm Plot",
            "Feature Clustering"
            ])
        with tp:
            st.image("shap29tp.png")
        with fp:
            st.image("shap29fp.png")
        with tn:
            st.image("shap29tn.png")
        with fn:
            st.image("shap29fn.png")
        with heatm:
            st.image("shap29heatm.png")
        with bees:
            st.image("shap29bees.png")
        with f_cluster:
            st.image("shap29f_clustering.png")
with tab3:
   ft_sixseven, ft_twonine = st.tabs(["67 Features", "29 Features"])
   with ft_sixseven:
        tp, fp, tn, fn= st.tabs([
            "True Positive", 
            "False Positive", 
            "True Negative", 
            "False Negative"
            ])
        with tp:
            st.image("lime67tp.png")
        with fp:
            st.image("lime67fp.png")
        with tn:
            st.image("lime67tn.png")
        with fn:
            st.image("lime67fn.png")
   with ft_twonine:
        tp, fp, tn, fn = st.tabs([
            "True Positive", 
            "False Positive", 
            "True Negative", 
            "False Negative"
            ])
        with tp:
            st.image("lime29tp.png")
        with fp:
            st.image("lime29fp.png")
        with tn:
            st.image("lime29tn.png")
        with fn:
            st.image("lime29fn.png")

with tab4:
    prem_67, prem_29 = st.tabs(['67 Features', '29 Features'])
    with prem_67:
        st.image("prem_feature_important_67.png")
    with prem_29:
        st.image("prem_feature_important_29.png")

with tab5:
    st.header("Team Members")
    st.markdown(team_members)