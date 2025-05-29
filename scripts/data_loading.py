import os
import pandas as pd
from pathlib import Path

def load_audio_filepaths(audio_dir: Path) -> pd.DataFrame:
    file_info = {
        'filename': [],
        'filepath': []
    }
    for file in os.listdir(audio_dir):
        file_info['filename'].append(file)
        file_info['filepath'].append(str(audio_dir / file))

    return pd.DataFrame(file_info)