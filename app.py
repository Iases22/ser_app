import streamlit as st
import requests
from PIL import Image


st.set_page_config(layout="wide")

# image = Image.open('Sersa_logo.jfif') #logo
# st.image(image, width = 340) #logo width

'''
# SERSA - Speech Emotion Recognizer & Song Advisor

'''


st.sidebar.header('About this project') #sidebar title
st.sidebar.markdown("**What is SERSA?**  \nSERSA was developed as a deep learning project to identify emotions from speech. SERSA takes a sample of speech as input, analyzes it based on thousands of previous examples of speech and returns the primary emotion it detected in the voice sample. Based on the ouput, SERSA then provides a list of songs scraped from the Spotify API which 'match' the emotion.")

st.sidebar.markdown("**Why is speech emotion recognition important?**  \nSpeech emotion recognition (SER) is notoriously difficult to do, not just for machines but also for us humans! The applications of SER are varied - from business (improving customer service), to healthcare (telemedicine and supporting people affected by alexithymia) to our personal lives.")
st.sidebar.markdown('''**What was our approach?** \nUsing a Multilayer Perceptron (MLP) Classifier we were able to train a model on the RAVDESS and TESS datasets and achieve XX percent accuracy. We also tried using CNN and RNN models but they were less effective.''' )

st.sidebar.markdown("*Sidenotes*:  \nEmotion recognition is an intrinsically subjective task (i.e. what one person considers angry another might consider sad). SERSA was trained on a specific set of voice samples and will therefore extrapolate based on those - thus, you may find SERSA's predictions to be odd at times - that's the nature of the game!")

st.subheader(":musical_note: Upload your voice recording here using .wav format")
uploaded_file = st.file_uploader("Select file from your directory")

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes)

