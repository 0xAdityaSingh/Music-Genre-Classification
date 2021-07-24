import librosa, IPython
from scipy.ndimage.measurements import variance
from numpy.core.fromnumeric import mean
import librosa.display
import sklearn
import matplotlib.pyplot as plt
from AudioFeaturizer.audio_featurizer import *

file = 'Data/genres_original/all/blues.00000.wav'

# print(audio_process(r'Data/genres_original/blues/blues.00000.wav'))


Signal , sr = librosa.load(file , sr = 22050) # n_samples = 2.6 * 60 * 22050

# # plt.figure(figsize=(15,5))
# # librosa.display.waveplot(Signal , sr = sr)
# # plt.xlabel('Time')
# # plt.ylabel('Amplitude')
# # plt.title("Classical music signal")
# # plt.show()

# # IPython.display.Audio(Signal, rate=sr)

# # X = librosa.stft(Signal)
# # Xdb = librosa.amplitude_to_db(abs(X))
# # plt.figure(figsize=(14, 5))
# # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
# # plt.title("Spectogram")
# # plt.colorbar()

spectral_centroids = librosa.feature.spectral_centroid(Signal, sr=sr)[0]
# #.spectral_centroid will return an array with columns equal to the number of frames present in your sample.

# # Computing the time variable for visualization
# # plt.figure(figsize=(15, 5))
# # frames = range(len(spectral_centroids))
# # t = librosa.frames_to_time(frames)

# # Normalising the spectral centroid for visualisation
hop_length = 512
chromagram = librosa.feature.chroma_stft(Signal, sr=sr, hop_length=hop_length)
def normalize(Signal, axis=0):
    return sklearn.preprocessing.minmax_scale(Signal, axis=axis)
mfccs20 = librosa.feature.mfcc(Signal, sr=sr, n_mfcc=19)
# #Plotting the Spectral Centroid along the waveform
# # librosa.display.waveplot(Signal, sr=sr, alpha=0.4)
# # plt.plot(t, normalize(spectral_centroids), color='r')

print("Chroma feature Mean: ",mean(chromagram))
print("Chroma feature Variance: ",variance(chromagram))

print("Spectral Centroids Mean: ",mean(spectral_centroids))
print("Spectral Centroids Variance: ",variance(spectral_centroids))
print("mfccs20 Mean: ",mean(mfccs20))
print("mfccs20 Variance: ",variance(mfccs20))

# # plt.show()