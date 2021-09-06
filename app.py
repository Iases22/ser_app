import streamlit as st
import requests

'''
# SERSA front

'''

st.subheader("Choose a mp3 file that you extracted from the work site")
uploaded_file = st.file_uploader("Select file from your directory")
if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format='audio/mp3')




# enter here the address of your flask api
url = ''

params = dict()

response = requests.get(url, params=params)

prediction = response.json()

pred = prediction['prediction']

pred
