# import librosa
# from librosa.feature.spectral import melspectrogram
# import matplotlib.pyplot as plt
# import librosa.display
# #from apps import sersa

# def create_spec():
#     #send uploaded_file
#     emotions = {
#         '01': 'neutral',
#         '02': 'calm',
#         '03': 'happy',
#         '04': 'sad',
#         '05': 'angry',
#         '06': 'fearful',
#         '07': 'disgust',
#         '08': 'surprised'
#         }

#     features = {
#         '01': 'MFCC',
#         '02': 'Mel-spectrogram',
#         '03': 'Chroma',
#         '04': 'Tempogram'
#     }

#     fig = plt.figure(figsize=(40, 20))
#     fig.subplots_adjust(hspace=0.4, wspace=0.4)
#     for k in features.keys():
#         fig.add_subplot(2, 2, int(k))
#         plt.title(features[k])
#         data, sample_rate = librosa.load('OAF_back_angry.wav')
#         if features[k] == 'MFCC':
#             f = librosa.feature.mfcc(y=data, sr=sample_rate)
#         if features[k] == 'Mel-spectrogram':
#             f = librosa.feature.melspectrogram(y=data, sr=sample_rate)
#         if features[k] == 'Chroma':
#             f = librosa.feature.chroma_stft(y=data, sr=sample_rate)
#         if features[k] == 'Tempogram':
#             f = librosa.feature.tempogram(y=data, sr=sample_rate )
#         librosa.display.specshow(f, sr=sample_rate, x_axis='time')
#     return fig
