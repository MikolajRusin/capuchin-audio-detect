from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass
import numpy as np
import librosa

N_SAMPLES = 72000

def fix_audio_length(wav: np.ndarray) -> np.ndarray:
    wav = wav[:N_SAMPLES]
    zero_padding = np.zeros(N_SAMPLES - wav.shape[0], dtype=np.float32)
    wav = np.concatenate((wav, zero_padding))

    return wav

def get_MFCC(wav: np.ndarray, n_mfcc: int, expand_dims: bool=False) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=wav, sr=24000, n_fft=512, hop_length=128, n_mfcc=n_mfcc)
    if expand_dims:
        mfcc = np.expand_dims(mfcc, axis=2)
    
    return mfcc

def MFCC_mean_std(mfcc):
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    feature_vector = np.concatenate([mfcc_mean, mfcc_std])

    return feature_vector

@dataclass
class AudioMFCCExtractor(BaseEstimator, TransformerMixin):
    n_mfcc: int = 20
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        mfcc_features = []
        for wav in X:
            wav = fix_audio_length(wav)
            mfcc = get_MFCC(wav, n_mfcc=self.n_mfcc)
            feature_vector = MFCC_mean_std(mfcc)
            mfcc_features.append(feature_vector)

        return np.array(mfcc_features)