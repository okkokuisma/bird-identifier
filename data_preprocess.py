import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation
import scipy.signal as sig
import os
# import warnings
# warnings.filterwarnings("ignore")

def parse_species():
    species = pd.read_csv('/Users/okkokuisma/projektit/birds/occurence_data/taxon-export.tsv', sep='\t')
    species = species.dropna()
    species = species.sort_values('Havaintomäärä Suomesta', ascending=False).iloc[:260, :]
    return species

def load_occurrences(path = '/Volumes/COCO-DATA/0000764-250426092105405/occurrence.txt'):
    return pd.read_csv(path, sep='\t', skiprows=[5644])

def drop_occurrence_cols(occurrences, cols_to_keep = [ 'gbifID', 'species', 'decimalLatitude', 'decimalLongitude' ]):
    return occurrences.loc[:, cols_to_keep]

def filter_occurrences(occurrences, species):
    occurrences = occurrences[occurrences['species'].isin(species['Tieteellinen nimi'])]
    mask = (occurrences.groupby('species').count()['gbifID'] < 50) 
    dropped_species = occurrences.groupby('species').count()['gbifID'][mask]
    occurrences = occurrences.loc[~occurrences['species'].isin(dropped_species.index.to_list()), :]
    return occurrences

def get_occurrences():
    occurrences = load_occurrences()
    occurrences = drop_occurrence_cols(occurrences)
    occurrences = filter_occurrences(occurrences, species)
    return occurrences

def load_signal(path, sr = 48000):
    signal, sr = librosa.load(path = path, sr = sr)
    return signal

# The frequency range of most bird vocalizations is between 250 Hz and 8.3 kHz
def filter_freqs(signal, sr = 48000, f_highpass = 150, f_lowpass = 10000):
    nyquist = sr / 2.0
    b_hp, a_hp = sig.butter(4, f_highpass/nyquist, btype="highpass")
    b_lp, a_lp = sig.butter(4, f_lowpass/nyquist, btype="lowpass")
    signal_hp = sig.filtfilt(b_hp, a_hp, signal)
    signal = sig.filtfilt(b_lp, a_lp, signal_hp)
    return signal

def calculate_spectrogram(signal, n_fft=512, hop_length=384, win_length=512):
    spectrogram = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann"
    )
    return np.abs(spectrogram)

def separate_bird_song(spectrogram):
    spectrogram_max = np.max(spectrogram)
    spectrogram_normalized = spectrogram / spectrogram_max
    col_means = np.mean(spectrogram_normalized, axis=0)
    row_means = np.mean(spectrogram_normalized, axis=1)
    mask = (spectrogram_normalized >= 3 * col_means) & (spectrogram_normalized >= 3 * row_means[:, np.newaxis])
    masked_spectrogram = np.where(mask, 1, 0)
    be = binary_erosion(masked_spectrogram).astype(masked_spectrogram.dtype)
    bd = binary_dilation(be)
    ind_vec = np.where(np.any(bd, axis=0), 1, 0)
    ind_vec = binary_dilation(binary_dilation(ind_vec).astype(ind_vec.dtype))
    return spectrogram[:, ind_vec][:, :280]

def audio_preprocessing_pipeline(
    path,
    sr = 48000,
    f_highpass = 150,
    f_lowpass = 10000,
    n_fft = 512,
    hop_length = 384,
    win_length = 512
):
    signal = load_signal(path, sr)
    filtered_signal = filter_freqs(signal, sr, f_highpass, f_lowpass)
    spectrogram = calculate_spectrogram(filtered_signal, n_fft, hop_length, win_length)
    signal_segment = separate_bird_song(spectrogram)
    return signal_segment

def preprocess_and_save(occurrences):
    for i, id in enumerate(occurrences['gbifID'].values):
        if (i % 100 == 0):
            print('Processing file [%d]/[%d]\r'%(i, occurrences.shape[0]), end="")
        song_path = f"/Volumes/COCO-DATA/songs/{id}"
        npy_path = f"/Volumes/COCO-DATA/songs_npy/{id}.npy"
        try:
            if os.path.isfile(song_path):
                segment = audio_preprocessing_pipeline(song_path)
                np.save(npy_path, segment)
        except:
            continue

species = parse_species()
occurrences = get_occurrences()
preprocess_and_save(occurrences)