import streamlit as st
import streamlit.components.v1 as components
# from streamlit_player import st_player
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ser_app import spotify_query
# import webbrowser

st.set_page_config(layout="wide")

# # main page color
# CSS = """
# h2 {
#     color: #00008B;
# }
# h3 {
#     color: #00008B;
# }
# .stApp {
#     background-color: #D4B49D;
#     background-size: cover;
# }
# """
# st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

# # sidebar color
# st.markdown(
#     """
# <style>
# .css-17eq0hr {
#     background-color: #D4B49D;
# }
# </style>
# """,
#     unsafe_allow_html=True,
# )


'''
# SERSA - Speech Emotion Recognizer & Song Advisor
'''

# sidebar
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
######


st.subheader(
    ":musical_note: Upload your voice recording here using .wav format")
uploaded_file = st.file_uploader("Select file from your directory")

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes)


emoji_dict = {
    'calm': '😌',
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fearful': '😨',
    'disgust': '🤮'
}

url = 'https://emotion-ser2-k7ur66xaxa-ew.a.run.app/predict/'
#url = 'https://api-btzfftkewq-ew.a.run.app/predict/'

button = st.button('click to predict the emotion')

if button:
    # print is visible in the server output, not in the page
    print('button clicked!')

    files = {'file': audio_bytes}
    response_0 = requests.post(url, files=files)
    predicted_emotion = response_0.json()['emotion']['0']
    probas = response_0.json()['probability']

    #putting probas dictionary into a dataframe
    v = list(probas.values())
    c = list(probas.keys())
    predicted_probas = pd.DataFrame([v], columns=c)

    #creating a ranked dictionary and putting it into a dataframe
    sort_probas = sorted(probas.items(), key=lambda x: x[1], reverse=True)

    ranked_emotions = []
    ranked_values = []
    for i in sort_probas:
        ranked_emotions.append(i[0])
        ranked_values.append(i[1])

    ranked_predicted_probas = pd.DataFrame([ranked_values],
                                           columns=ranked_emotions)

    #picking out the predicted emotion and displaying it with an emoji
    if predicted_emotion=='disgust':
        emotion_word='disgusted'
    else:
        emotion_word=predicted_emotion

    st.header(f'SERSA thinks you\'re **{emotion_word}** ' + emoji_dict[predicted_emotion])

    """

    """

    #displaying predicted probabilities of each emotion as a bar chart
    reverse_ranked_emotions = ranked_emotions
    reverse_ranked_values = ranked_values
    reverse_ranked_emotions.reverse()
    reverse_ranked_values.reverse()

    fig, ax = plt.subplots(figsize=(8, 1))
    right_side = ax.spines["right"]
    top_side = ax.spines['top']
    right_side.set_visible(False)
    top_side.set_visible(False)

    ax.barh(reverse_ranked_emotions,
            reverse_ranked_values,
            color=['r', 'y', 'g', 'b', 'c', 'm'])

    ax.set_yticklabels(reverse_ranked_emotions, fontsize=5)
    ax.set_xticklabels(list(range(0, 100, 10)), fontsize=5)

    for index, value in enumerate(reverse_ranked_values):
        if value < 0.1:
            continue
        plt.text(value, index, str(value), fontsize=5)

    st.pyplot(fig)
    """

    """


    # recommending songs
    spotify_artist, spotify_tracknames, spotify_urls = spotify_query.get_spotify_links(
        predicted_emotion)

    st.subheader('Recommended songs:')
    for i in range(5):
        st.markdown(
            f"[{spotify_artist[i]} - {spotify_tracknames[i]}]({spotify_urls[i]})"
        )
        components.iframe(spotify_urls[i])
