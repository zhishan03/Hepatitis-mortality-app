# Load Core Pks
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# Utils
import os
import joblib
import hashlib
import base64

# Data Viz Pks
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
matplotlib.use('Agg')
from PIL import Image
import seaborn as sns

# DB
from managed_db import *

feature_names_best = ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']

gender_dict = {"male":1, "female":2}
feature_dict = {"No":1, "Yes":2}

symptoms = """ - Fatigue
            - Flu-like symptoms 
            - Dark urine 
            * Pale stool 
            * Abdominal Pain 
            * Loss of appetite 
            * Unexplained weight loss 
            * Yellow skin and eyes, which may be signs of jaundice
        """

html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Disease Mortality Prediction </h1>
		<h5 style="color:white;text-align:center;">A Web App by Joy Chen </h5>
		</div>
		"""

descriptive_message_temp ="""
	<div style="background-color:#EEEEEE;overflow-x: auto; padding:10px;border-radius:5px;">
		<h3 style="text-align:left;color:black;padding:10px">Definition</h3>
		<p style=padding:10px>Hepatitis means inflammation of the liver. The liver is a vital organ that processes nutrients, filters the blood, and fights infections. When the liver is inflamed or damaged, its function can be affected. Heavy alcohol use, toxins, some medications, and certain medical conditions can cause hepatitis.</p>
	</div>
	"""

# Password
def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hashes(password,hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    return False

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

def get_key(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return key

def get_fvalue(val):
    feature_dict = {"No":1, "Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

# Load ML model
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

import lime
import lime.lime_tabular

def main():
    st.markdown(html_temp.format('#464660'),unsafe_allow_html=True)
    st.text('\n')

    menu = ["Home","Plot","Prediction"]

    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.write("""### What is Hepatitis❓""")
        st.markdown(descriptive_message_temp,unsafe_allow_html=True)
        st.text('\n \n')
        # image = Image.open('hepatitisC.gif')
        # st.image(image, caption='Hepatitis Virus')
        file_ = open("hepatitisC.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.text('\n \n')
        st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">', unsafe_allow_html=True)
        st.text('\n')
        st.write("---")
        st.subheader("Common Symptoms of hepatitis:")
        """ 
            - Fatigue
            - Flu-like symptoms 
            - Dark urine 
            - Pale stool 
            - Abdominal Pain 
            - Loss of appetite 
            - Unexplained weight loss 
            - Yellow skin and eyes, which may be signs of jaundice
        """
        st.write(
            """If you would like to read up more on hepatitis, please go on to [this article](https://www.cdc.gov/hepatitis/abc/index.htm) written 
               by Centers for Disease Control and Prevention!""")
    elif choice == "Plot":
        st.text('\n')
        st.write("""Analyzing the [hepatitis data set](https://archive.ics.uci.edu/ml/datasets/hepatitis) in **UCI's Machine Learning Repository** ✍️""")
        st.subheader("Data Vis Plot")
        df = pd.read_csv("data/clean_hepatitis_dataset.csv")
        if "page" not in st.session_state:
            st.session_state.page = 0

        def next_page():
            st.session_state.page += 1

        def prev_page():
            st.session_state.page -= 1

        col1, col2, col3, _ = st.columns([0.1, 0.17, 0.1, 0.63])

        if st.session_state.page < 4:
            col3.button(">", on_click=next_page)
        else:
            col3.write("")  # this makes the empty column show up on mobile

        if st.session_state.page > 0:
            col1.button("<", on_click=prev_page)
        else:
            col1.write("")  # this makes the empty column show up on mobile

        col2.write(f"Page {1+st.session_state.page} of {5}")
        start = 10 * st.session_state.page
        end = start + 10
        st.write("")
        st.write(df.iloc[start:end])

        st.write("---")
        st.subheader("Patient Outcome")
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(x=df['class'],
                    y=df['index'], color='goldenrod', ax=ax, ci=None)
        ax.set_xlabel('Outcome')
        ax.set_ylabel('Patient Count')
        st.pyplot(fig)
        st.markdown("""Out of the grand total of **155** patients with  hepatitis, **124** survived! On the other hand, unfortunately
        **31** patients didn't live past this disease. We will take a quick look at why.""")

        st.write("---")
        # Pie Plot
        labels= ["Less than 10", "10-20","20-30","30-40","40-50","50-60","60-70","70 and more"]
        bins = [0,10,20,30,40,50,60,70,80]
        freq_df = df.groupby(pd.cut(df['age'],bins=bins,labels=labels)).size()
        freq_df = freq_df.reset_index(name='count')
        labels = ['20-30', '30-40','40-50','50-60','all other age']
        sizes = [18.7, 32.3, 22.6, 15.5, 10.9]
        explode = (0, 0.1, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)
        st.markdown("""Highest prevalence of Hepatitis is from **30-40** group followed by **40-50** group. 
        The least is individuals under **10** group, and elderly above **70** group""")

        st.write("---")
        st.subheader("Patient's Albumin Level by Age")
        fig = plt.figure()
        ax = fig.subplots()
        g = sns.scatterplot(x=df['albumin'],y=df['age'],hue=df['sex'],palette=['green','red'],data=df)
        ax.set_xlabel('Age')
        ax.set_ylabel('Albumin')
        plt.legend(labels=["Female","Male"], title="Sex", fontsize = '8', title_fontsize = "9")
        st.pyplot(fig)
        st.markdown("""A high level of albumin is most likely to occur in age between **20 to 60**. This means that patients
                        within such an age group has a higher probability of dying due to hepatitis""")

        st.write("---")
        st.subheader("Sorted Feature Score")
        df = pd.read_csv("data/data.csv")
        st.dataframe(df)
        st.markdown("""Judging from the F-scores, **protime** (the time it takes for a clot to form in a blood sample) ranks the highest
        in its descriminative power followed by **sgot**, **bilirubin**, and **age**""")

        st.write("---")
        if st.checkbox("Area Chart"):
            all_columns = df.columns.to_list()
            feat_choices = st.multiselect("Choose a Feature",all_columns)
            new_df = df[feat_choices]
            st.area_chart(new_df)

    else:
        st.subheader("Predictive Analytics")
        st.markdown("""Enter stats below to predict the life/death rate by hepatitis""")
        age = st.number_input("Age",7,80)
        sex = st.radio("Sex",tuple(gender_dict.keys()))
        steroid = st.radio("Do You Take Steroids?",tuple(feature_dict.keys()))
        antivirals = st.radio("Do You Take Antivirals?",tuple(feature_dict.keys()))
        fatigue = st.radio("Do You Have Fatigue",tuple(feature_dict.keys()))
        spiders = st.radio("Presence of Spider Naeve",tuple(feature_dict.keys()))
        ascites = st.selectbox("Ascities",tuple(feature_dict.keys()))
        varices = st.selectbox("Presence of Varices",tuple(feature_dict.keys()))
        bilirubin = st.number_input("bilirubin Content",0.0,8.0)
        alk_phosphate = st.number_input("Alkaline Phosphate Content",0.0,296.0)
        sgot = st.number_input("Sgot",0.0,648.0)
        albumin = st.number_input("Albumin",0.0,6.4)
        protime = st.number_input("Prothrombin Time",0.0,100.0)
        histology = st.selectbox("Histology",tuple(feature_dict.keys()))
        feature_list = [age,get_value(sex,gender_dict),get_fvalue(steroid),get_fvalue(antivirals),get_fvalue(fatigue),get_fvalue(spiders),get_fvalue(ascites),get_fvalue(varices),bilirubin,alk_phosphate,sgot,albumin,int(protime),get_fvalue(histology)]

        st.write(feature_list)
        pretty_result = {"age":age,"sex":sex,"steroid":steroid,"antivirals":antivirals,"fatigue":fatigue,"spiders":spiders,"ascites":ascites,"varices":varices,"bilirubin":bilirubin,"alk_phosphate":alk_phosphate,"sgot":sgot,"albumin":albumin,"protime":protime,"histolog":histology}
        st.json(pretty_result)
        single_sample = np.array(feature_list).reshape(1,-1)

        # ML
        model_choice = st.selectbox("Select Model", ["LR","KNN","DecisionTree"])
        if st.button('Predict'):
            if model_choice == "KNN":
                loaded_model = load_model("models/knn_hepB_model.pkl")
                prediction = loaded_model.predict(single_sample)
                pred_prob = loaded_model.predict_proba(single_sample)
            elif model_choice == "LR":
                loaded_model = load_model("models/logistic_regression_hepB_model.pkl")
                prediction = loaded_model.predict(single_sample)
                pred_prob = loaded_model.predict_proba(single_sample)
            else:
                loaded_model = load_model("models/decision_tree_clf_hepB_model.pkl")
                prediction = loaded_model.predict(single_sample)
                pred_prob = loaded_model.predict_proba(single_sample)

            st.write(prediction)
            if prediction == 1:
                st.warning("Patient Dies")
            else:
                st.success("Patient Lives")
                pred_probability_score = {"Die":pred_prob[0][0]*100, "Live":pred_prob[0][1]*100}
                st.subheader("Prediction Probability Score using {}".format(model_choice))
                st.json(pred_probability_score)
            
        if st.checkbox("Interpret"):
            if model_choice == "KNN":
                loaded_model = load_model("models/knn_hepB_model.pkl")
            elif model_choice == "DecisionTree":
                loaded_model = load_model("models/decision_tree_clf_hepB_model.pkl")
            else:
                loaded_model = load_model("models/logistic_regression_hepB_model.pkl")
                                

            # loaded_model = load_model("models/logistic_regression_model.pkl")							
            # 1 Die and 2 Live
            df = pd.read_csv("data/clean_hepatitis_dataset.csv")
            x = df[['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']]
            feature_names = ['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']
            class_names = ['Die(1)','Live(2)']
            explainer = lime.lime_tabular.LimeTabularExplainer(x.values,feature_names=feature_names, class_names=class_names,discretize_continuous=True)
            # The Explainer Instance
            exp = explainer.explain_instance(np.array(feature_list), loaded_model.predict_proba,num_features=13, top_labels=1)
            exp.show_in_notebook(show_table=True, show_all=False)
            # exp.save_to_file('lime_oi.html')
            st.write(exp.as_list())
            new_exp = exp.as_list()
            label_limits = [i[0] for i in new_exp]
            # st.write(label_limits)
            label_scores = [i[1] for i in new_exp]
            plt.barh(label_limits,label_scores)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            plt.figure(figsize=(20,10))
            fig = exp.as_pyplot_figure()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

if __name__ == '__main__':
    main()

