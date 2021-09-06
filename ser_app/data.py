########experimental - delete this function later#######

import streamlit as st
import requests
import joblib
import pandas as pd
import ser
from ser.data import extract_features



#Extract features (mfcc, chroma, mel, temp) from a sound file
def extract_features(sound_file, mfcc, chroma, mel, temp):
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

import streamlit as st
import requests
import joblib
import pandas as pd
import ser
from ser.data import extract_features

###for testing purposes until we get the api
#Load an audio file and transform it
def file_handler(audio_bytes):
    with open("pip.wav", "wb") as file:
        file.write(audio_bytes)
    observed_emotions = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']
    x_pred_preprocessed = x_pred_preprocessing('pip.wav')
    prediction = return_predict(x_pred_preprocessed)
    predicted_probas = predict_proba(observed_emotions, x_pred_preprocessed)
    return prediction, predicted_probas
    ###delete pip.wav




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
########



# def extract_features(sound_file, mfcc, chroma, mel, temp):
#     sample_rate = sound_file.samplerate
#     if chroma:
#         stft = np.abs(librosa.stft(sound_file))
#     result = np.array([])
#     if mfcc:
#         mfccs = np.mean(librosa.feature.mfcc(y=sound_file, sr=sample_rate, n_mfcc=40).T,
#                         axis=0)
#         result = np.hstack((result, mfccs))
#     if chroma:
#         chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
#                          axis=0)
#         result = np.hstack((result, chroma))
#     if mel:
#         mel = np.mean(librosa.feature.melspectrogram(sound_file, sr=sample_rate).T,
#                       axis=0)
#         result = np.hstack((result, mel))
#     if temp:
#         temp = np.mean(librosa.feature.tempogram(y=sound_file, sr=sample_rate).T,
#                        axis=0)
#         result = np.hstack((result, temp))
#     return result
