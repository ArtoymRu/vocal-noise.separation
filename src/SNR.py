import numpy as np
import librosa

def calculate_snr(original_file, processed_file):
    """
    Calculate the Signal-to-Noise Ratio (SNR).
    
    :param original_file: Path to the original audio file.
    :param processed_file: Path to the processed (voice extracted) audio file.
    :return: SNR in decibels.
    """
    # Load audio files
    original, sr_orig = librosa.load(original_file, sr=None)
    processed, sr_proc = librosa.load(processed_file, sr=None)
    '''
    # Ensure same sample rate
    if sr_orig != sr_proc:
        raise ValueError("Sample rates do not match")

    '''
    # Truncate longer audio to match the shorter one
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]

    # Calculate SNR
    noise = original - processed
    snr = 10 * np.log10(np.sum(processed ** 2) / np.sum(noise ** 2))
    return snr