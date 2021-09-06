import streamlit as st
import requests

'''
# SERSA - Speech Emotion Recognizer & Song Advisor

'''

st.subheader("Upload your voice recording here")
uploaded_file = st.file_uploader("Select file from your directory")
if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes)




# enter here the address of your flask apiUpl
url = ''

params = dict()

response = requests.get(url, params=params)

prediction = response.json()

pred = prediction['prediction']

pred
