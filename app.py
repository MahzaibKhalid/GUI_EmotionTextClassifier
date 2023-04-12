import streamlit as st
import altair as alt
import plotly.express as px
from PIL import Image

import pandas as pd
import numpy as np
from datetime import datetime


import pickle

pipe_lr = pickle.load(open("model\Emotion_Classifier.pkl",'rb'))

#Functions 

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results= pipe_lr.predict_proba([docx])
    return results

#Emoji Dictionary:

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


def main():
    image = Image.open('image\mainn.png')
    st.image(image, caption=None, width=None, use_column_width=None)
    st.title("Emotion from Text Classifier App")
    
    menu =["Home","Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.subheader("Home-Emotion in Text")
        
    elif choice == "Monitor":
        st.subheader("Monitor App")
        with st.form(key="emotion_clf_form"):
            raw_text =st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")
        if submit_text:
            col1,col2 = st.columns(2)
            
            #appky functions here
            predicition = predict_emotions(raw_text)
            probability =get_prediction_proba(raw_text)

            with col1:
                
                st.success("Original Text")
                st.write(raw_text)
                
                st.success("Predcition")
                emoji_icon = emotions_emoji_dict[predicition]
                st.write("{}:{}".format(predicition,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))
                
            with col2:
                st.success("Predicition Probability")
                
                st.write(probability)               
                proba_df=pd.DataFrame(probability,columns=pipe_lr.classes_)
                st.write(proba_df.T)
                proba_df_clean=proba_df.T.reset_index()
                proba_df_clean.columns=["emotions","probability"]
                
                fig=alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                st.altair_chart(fig,use_container_width=True)
                
    else:
        st.header("About")
        st.success("Emotion Detection is technique in which we can detect emotions like\
                     angry,fear,happy,hate and sad. In this project we have used Machine Learning\
                         classifiers to build a model.")

        
        
        
if __name__ == '__main__':
    main()