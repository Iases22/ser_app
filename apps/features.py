import streamlit as st
import matplotlib.pyplot as plt
from ser_app.spectrogram import create_spec

def app():
    st.title('Sound Features')
    st.write('Features used in the project in spectrogram:')
    spec = create_spec()
    st.pyplot(spec)
