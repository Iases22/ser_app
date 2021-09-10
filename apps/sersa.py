import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ser_app import spotify_query
import base64

def app():

    # main page and sidebar background images
    main_bg = "background5.jpg"
    main_bg_ext = "jpg"
    side_bg = "background4.jpg"
    side_bg_ext = "jpg"
    st.markdown(f"""
        <style>
        .reportview-container {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
        }}
        .css-17eq0hr {{
            background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
        }}
        </style>
        """,
                unsafe_allow_html=True)

    '''
    # SERSA - Speech Emotion Recognizer & Song Advisor
    '''
    st.title('SERSA - Speech Emotion Recognizer & Song Advisor')
    # st.header('*She’s Everyone’s Reliable Song Assistant!*')
    # st.write('')

    # upload file
    audio_bytes = None
    st.subheader(
        ":musical_note: Upload your voice recording here using .wav format")
    uploaded_file = st.file_uploader("Select file from your directory")

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes)

    # make prediction
    emoji_dict = {
        'calm': '😌',
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fearful': '😨',
        'disgust': '🤮'
    }

    url = 'https://emotion-ser2-k7ur66xaxa-ew.a.run.app/predict/'

    button = st.button('click to predict the emotion')

    if button:
        # print is visible in the server output, not in the page
        print('button clicked!')

        if audio_bytes is not None:

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
            if predicted_emotion == 'disgust':
                emotion_word = 'disgusted'
            else:
                emotion_word = predicted_emotion

            st.header(f'SERSA thinks you\'re **{emotion_word}** ' +
                    emoji_dict[predicted_emotion])
            st.write('')
            """

            """

            #displaying predicted probabilities of each emotion as a bar chart
            reverse_ranked_emotions = ranked_emotions
            reverse_ranked_values = ranked_values
            reverse_ranked_emotions.reverse()
            reverse_ranked_values.reverse()

            fig, ax = plt.subplots(figsize=(8, 1))
            right_side = ax.spines['right']
            top_side = ax.spines['top']
            right_side.set_visible(False)
            top_side.set_visible(False)

            ax.barh(reverse_ranked_emotions,
                    reverse_ranked_values,
                    color=['r', 'y', 'g', 'b', 'c', 'm'])

            ax.set_yticklabels(reverse_ranked_emotions, fontsize=5)
            plt.xticks(fontsize=5)

            for index, value in enumerate(reverse_ranked_values):
                if value < 0.1:
                    continue
                plt.text(value, index, str(value), fontsize=5)

            st.pyplot(fig)
            st.write('')
            """

            """

            # recommending songs
            spotify_artist, spotify_tracknames, spotify_urls = spotify_query.get_spotify_links(
                predicted_emotion)

            st.subheader('Recommended songs:')
            for i in range(5):
                # st.markdown(
                #     f"[{spotify_artist[i]} - {spotify_tracknames[i]}]({spotify_urls[i]})"
                # )
                track_url = spotify_urls[i]
                sul = track_url.split('/')
                embed_url = sul[0] + '//' + sul[2] + '/embed/' + sul[
                    3] + '/' + sul[4]
                components.iframe(embed_url)

        else:
            st.write('You need to upload a file!')
