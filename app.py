import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import os

# profiling
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# Machine Learning
import sklearn
from pycaret.regression import setup, compare_models, pull, save_model

PATH = r'sorcedata.csv'




def run_app():
    """
    Here is MLbox API
    """
    try:
        with st.sidebar:
            st.image(
                "https://techyon.intervieweb.it/immagini/login/img_17_big_data_vector.png")
            st.title("AutoML Service")
            choice = st.radio(
                "Navigation", ["Upload", "Profiling", "ML", "Download"])
            st.info('This application allows you to build an automated ML pipline using Streamlit, Pandas Profiling and PySpark. And it dawnright magic!')

        if os.path.exists(PATH):
            df = pd.read_csv(PATH, index_col=None)
        if choice == "Upload":
            st.title("Upload Your Data for Modeling!")
            file = st.file_uploader("Upload your DataSet Here")
            if file:
                df = pd.read_csv(file, index_col=[0])
                df.to_csv(PATH, index=None)
                st.dataframe(df)

        if choice == "Profiling":
            st.title("Automated Exploratory Data Analysis")
            profile_report = df.profile_report()
            st_profile_report(profile_report)

        if choice == "ML":
            st.title("Machine Learning go BRR...")
            chosen_target = st.selectbox('Choose the Target Column', df.columns)
            if st.button('Run Modelling'):
                    setup(df, target=chosen_target, silent=True, fold_shuffle=True)
                    setup_df = pull()
                    st.dataframe(setup_df)
                    best_model = compare_models()
                    compare_df = pull()
                    st.dataframe(compare_df)
                    save_model(best_model, 'best_model')

        if choice == "Download":
            with open("best_model.pkl", 'rb') as f:
                st.download_button("Download Model", f, "best_model_test.pkl")
    except Exception:
        st.write("Sorry buddy, not today :(")


if __name__ == "__main__":
    run_app()
