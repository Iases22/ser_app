import streamlit as st
import requests
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import soundfile
import librosa
import ser
from ser.data import extract_features


########just for testing until we get the api -- will be removed later#########
def extract_features(file_name, mfcc, chroma, mel, temp):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X,
                                                 sr=sample_rate,
                                                 n_mfcc=40).T,
                            axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft,
                                                         sr=sample_rate).T,
                             axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,
                          axis=0)
            result = np.hstack((result, mel))
        if temp:
            temp = np.mean(librosa.feature.tempogram(y=X, sr=sample_rate).T,
                           axis=0)
            result = np.hstack((result, temp))
    return result

#Load an audio file and transform it
def x_pred_preprocessing(audio_path):
    x_pred_preprocessed = extract_features(audio_path,
                                           mfcc=True,
                                           chroma=False,
                                           mel=True,
                                           temp=True)
    x_pred_preprocessed = x_pred_preprocessed.reshape(1, 552)
    return x_pred_preprocessed


#Predict the emotion
def return_predict(x_pred_preprocessed, model_path='MLP_model.joblib'):
    model = joblib.load(model_path)
    prediction = model.predict(x_pred_preprocessed)
    return prediction[0]


#Return a dataframe giving the predicted probabilities for each emotion in observed_emotions
def predict_proba(observed_emotions, x_pred_preprocessed, model_path='MLP_model.joblib'):
    model = joblib.load(model_path)
    emotion_list = observed_emotions
    emotion_list.sort()
    model_pred_prob = pd.DataFrame((model.predict_proba(x_pred_preprocessed) * 100).round(2),
                                columns=emotion_list)
    return model_pred_prob
######################


'''
# SERSA - Speech Emotion Recognizer & Song Advisor

'''

st.subheader("Upload your voice recording here")
uploaded_file = st.file_uploader("Select file from your directory")
if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes)

    with open("pip.wav", "wb") as file:   #######to be removed and added to package (predict.py) later
        file.write(audio_bytes)           #######



# url = ''

button = st.button('click to predict the emotion')

if button:
    # print is visible in the server output, not in the page
    print('button clicked!')

    # response = request.post(url, audio_bytes)
    # response.json()

    observed_emotions = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']
    x_pred_preprocessed = x_pred_preprocessing('pip.wav')
    prediction = return_predict(x_pred_preprocessed)
    st.write(prediction)
    predicted_probas = predict_proba(observed_emotions, x_pred_preprocessed)

    hpp = predicted_probas.assign(hack='').set_index('hack')
    st.write(hpp)
    st.bar_chart(predicted_probas)


    # fig = predicted_probas.plot.pie(subplots=True)
    # st.pyplot(fig)



    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    # labels = observed_emotions   #predicted_probas.columns
    # st.write(labels)
    # pp_list = predicted_probas.values.tolist()
    # sizes = pp_list
    # st.write(sizes)
    # explode = (0.1, 0.1, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes,
    #         explode=explode,
    #         labels=labels,
    #         autopct='%1.1f%%',
    #         shadow=True,
    #         startangle=90)
    # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # st.pyplot(fig1)
