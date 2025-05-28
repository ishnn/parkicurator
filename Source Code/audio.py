import librosa
import numpy as np
import xgboost as xgb

def extract_audio_features(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file)

    # Extract features
    features = []

    # MDVP:Fo(Hz) - Fundamental frequency
    fo = librosa.yin(y, fmin=50, fmax=400)
    features.append(np.mean(fo))

    # MDVP:Flo(Hz) - Lowest frequency of the acoustic signal
    flo = np.min(fo)
    features.append(flo)

    # MDVP:Jitter(Abs) - Absolute jitter
    jitter = librosa.effects.split(y)
    jitter_abs = np.mean(librosa.effects.split(y))
    features.append(jitter_abs)

    # MDVP:Shimmer - Shimmer
    shimmer = librosa.effects.split(y)
    shimmer_mean = np.mean(shimmer)
    features.append(shimmer_mean)

    # MDVP:Shimmer(dB) - Shimmer in dB
    shimmer_db_mean = np.mean(librosa.feature.rms(y=y))
    features.append(shimmer_db_mean)

    # Shimmer:APQ3 - Amplitude Perturbation Quotient
    apq3 = np.mean(librosa.feature.rms(y=y))
    features.append(apq3)

    # Shimmer:APQ5 - Amplitude Perturbation Quotient
    apq5 = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features.append(apq5)

    # MDVP:APQ - Amplitude Perturbation Quotient
    apq = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features.append(apq)

    # Shimmer:DDA - Shimmer (in DDA units)
    dda = np.mean(librosa.feature.spectral_flatness(y=y))
    features.append(dda)

    # HNR - Harmonic-to-Noise Ratio
    hnr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    features.append(hnr)

    # RPDE - Recurrence Period Density Entropy
    rpde = np.mean(librosa.feature.tempogram(y=y, sr=sr))
    features.append(rpde)

    # spread1
    spread1 = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    features.append(spread1)

    # spread2
    spread2 = np.mean(librosa.feature.spectral_flatness(y=y))
    features.append(spread2)

    # D2
    d2 = np.mean(librosa.feature.melspectrogram(y=y, sr=sr))
    features.append(d2)

    # PPE
    ppe = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    features.append(ppe)

    return features


