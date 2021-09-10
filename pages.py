"""Keeps track of multiple Streamlit applications as multipages.
"""
import streamlit as st
import base64


class MultiApp:
    def __init__(self):
        #constructor to generate list of apps as pages
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    def run(self):
        app = st.set_page_config(layout="wide")
        app = st.sidebar.radio('Page Navigation',
                               self.apps,
                               format_func=lambda app: app['title'])
        app['function']()

        # sidebar style
        # side_bg = "background4.jpg"
        # side_bg_ext = "jpg"
        # st.markdown(f"""
        #     <style>
        #     .css-17eq0hr {{
        #         background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
        #     }}
        #     </style>
        #     """,
        #             unsafe_allow_html=True)

        # sidebar content
        st.sidebar.header('About this project')  #sidebar title
        st.sidebar.markdown(
            "**What is SERSA?**  \nSERSA was developed as a deep learning project to identify emotions from speech. SERSA takes a sample of speech as input, analyzes it based on thousands of previous examples of speech and returns the primary emotion it detected in the voice sample. Based on the ouput, SERSA then provides a list of songs scraped from the Spotify API which 'match' the emotion."
        )

        st.sidebar.markdown(
            "**Why is speech emotion recognition important?**  \nSpeech emotion recognition (SER) is notoriously difficult to do, not just for machines but also for us humans! The applications of SER are varied - from business (improving customer service), to healthcare (telemedicine and supporting people affected by alexithymia) to our personal lives."
        )
        st.sidebar.markdown(
            '''**What was our approach?** \nUsing a Multilayer Perceptron (MLP) Classifier we were able to train a model on the RAVDESS and TESS datasets and achieve over 90 percent accuracy. We also tried using CNN and RNN models but they were less effective.'''
        )

        st.sidebar.markdown(
            "*Sidenotes*:  \nEmotion recognition is an intrinsically subjective task (i.e. what one person considers angry another might consider sad). SERSA was trained on a specific set of voice samples and will therefore extrapolate based on those - thus, you may find SERSA's predictions to be odd at times - that's the nature of the game!"
        )
