import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pickle
import librosa
import math
import argparse
from scripts.feature_extractor import AudioMFCCExtractor
from sklearn.pipeline import Pipeline
from tqdm import tqdm

PROJECT_PATH = Path(__file__).resolve().parent.parent   # Main project path
MODELS_DIR = PROJECT_PATH / 'models'
MODEL_PIPE_PATH = MODELS_DIR / 'knn_pca_model_pipe.pkl'
SR = 24000
WINDOW = 72000
HOP = WINDOW

def sliding_window(audio: np.ndarray, window: int, hop: int, batch_size: int):
    n_windows = math.ceil((len(audio) - window) / hop) + 1
    
    batch = []
    for n in range(n_windows):
        slided_window = audio[int(n*hop):int(n*hop+window)]

        if slided_window.shape[0] < window:
            zero_padding = np.zeros(window - slided_window.shape[0], dtype=np.float32)
            slided_window = np.concatenate((slided_window, zero_padding), axis=0)

        batch.append(slided_window)
        if len(batch) == batch_size:
            yield np.array(batch)
            batch = []
    
    if len(batch) != 0:
        yield np.array(batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, help='Path to audio to be processed')
    parser.add_argument('--batch_size', type=int, help='Size of one batch')
    args = parser.parse_args()

    audio_path = args.audio_path
    batch_size = args.batch_size

    with open(MODEL_PIPE_PATH, 'rb') as f:
        model_pipe = pickle.load(f)

    scaler_pca_clf = model_pipe.steps[1:]
    new_pipe = Pipeline([
        ('mfcc', AudioMFCCExtractor()),
        *scaler_pca_clf
    ])

    audio, sr = librosa.load(audio_path, sr=SR, mono=True)
    results = []
    n_windows = math.ceil((((len(audio) - WINDOW) / HOP) + 1) / batch_size)
    for window_batch in tqdm(sliding_window(audio, WINDOW, HOP, batch_size=batch_size), total=n_windows):
        y_pred = new_pipe.predict(window_batch)
        results.append(y_pred)

    results = np.concatenate(results, axis=0)
    changes_idx = np.where(np.diff(results) != 0)[0] + 1
    changes_idx = np.insert(changes_idx, 0, 0)

    n_results = np.sum(results[changes_idx])
    detection_times = changes_idx[results[changes_idx] == 1] * (HOP / SR)

    print('Number of Detected capuchinbird')
    print(n_results)
    print("Detection times s")
    print(detection_times)

if __name__ == '__main__':
    main()