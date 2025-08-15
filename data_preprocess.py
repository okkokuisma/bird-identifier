import numpy as np
import pandas as pd
import librosa
from scipy.ndimage import binary_erosion, binary_dilation
import scipy.signal as sig
from os import path
import uuid
import torchaudio.transforms as T
import torch
from occurrence_data import get_occurrence_count_by_species, load_filtered_occurrences

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
    return spectrogram

def separate_bird_song(spectrogram):
    spectrogram_abs = np.abs(spectrogram)
    spectrogram_max = np.max(spectrogram_abs)
    spectrogram_normalized = spectrogram_abs / spectrogram_max
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

def preprocess_and_save(occurrences, song_dir_path = '/Volumes/COCO-DATA/songs/', npy_dir_path = '/Volumes/COCO-DATA/songs_npy/'):
    for i, id in enumerate(occurrences['gbifID'].values):
        song_path = path.join(song_dir_path, str(id))
        npy_path = path.join(npy_dir_path, str(id) + '.npy')
        try:
            if not path.isfile(npy_path):
                segment = audio_preprocessing_pipeline(song_path)
                np.save(npy_path, np.abs(segment))
        except:
            continue

        if (i % 100 == 0):
            print('Processed file [%d]/[%d]\r'%(i, occurrences.shape[0]), end="")

def augment_song(signal):
    time_stretch = T.TimeStretch(hop_length = 384, n_freq=257, fixed_rate=1.2)
    freq_masking = T.FrequencyMasking(freq_mask_param=80)
    time_masking = T.TimeMasking(time_mask_param=80)

    stretched = time_stretch(torch.tensor(signal))
    f_masked = freq_masking(np.abs(stretched))
    t_masked = time_masking(f_masked)

    return t_masked

def save_augmented(augmented_song, dir_path = '/Volumes/COCO-DATA/songs_npy/'):
    new_id = uuid.uuid1().int
    file_path = dir_path + f"{new_id}.npy"
    np.save(file_path, augmented_song)
    return new_id

def create_augmented_data_points(file_path = '/Volumes/COCO-DATA/0000764-250426092105405/augmented_occurrences.parquet', song_dir_path = '/Volumes/COCO-DATA/songs'):
    occurrences = load_filtered_occurrences()
    num_occurrences_by_species = get_occurrence_count_by_species(occurrences)
    diff = (1000 - num_occurrences_by_species) # at least 1000 data points for every species
    new_data_points = []

    for s, d in zip(diff.index, diff.values):
        if d > 0:
            species_occurrences = occurrences.loc[occurrences['species'] == s, :]['gbifID']
            occurrence_sample = species_occurrences.sample(d, replace=True).values

            for id in occurrence_sample:
                try:
                    song_path = path.join(song_dir_path, str(id))
                    signal = audio_preprocessing_pipeline(song_path)
                    augmented = augment_song(signal)
                    augmented_id = save_augmented(augmented)
                    new_data_points.append((augmented_id, s))
                except:
                    continue

    augmented_data_points = pd.DataFrame(new_data_points)
    augmented_data_points.to_parquet(file_path)
