import os
import numpy as np
from numba import njit
import librosa
from matplotlib import pyplot as plt

@njit
def swaraRange(pitch, bandwidth=0, refPitch=69, refFreq=440):
    """Computes the lower and upper frequency/ies of a MIDI pitch based on shurti scale


    Args:
        pitch (int): MIDI pitch value
        bandwidth (int): Tolerance for the upper and lower frequency values of a given pitch 
        refPitch (int): Reference pitch (Tonic) (default: 69)
        refFreq (float): Frequency of reference pitch (Tonic) (default: 440.0)

    Returns:
        lower (float): Lower value of the pitch frequency adjusted with bandwidth
        upper (float): Upper value of the pitch frequency adjusted with bandwidth
    """
    shruti_ratios = [1, 256 / 243, 16 / 15, 10 / 9, 9 / 8, 32 / 27,
                     6 / 5, 5 / 4, 81 / 64, 4 / 3, 27 / 20, 45 / 32, 729 / 512,
                     3 / 2, 128 / 81, 8 / 5, 5 / 3, 27 / 16, 16 / 9, 9 / 5, 15 / 8,
                     243 / 128]

    shifted_freq_ref = refFreq
    diff = pitch - refPitch
    divisor = int(abs(diff) / 12)
    index = abs(diff) % 12
    if pitch < refPitch:
        if index == 0:
            bumpUp = 0
        else:
            bumpUp = 1
        index = (12 - index) % 12
        shifted_freq_ref = refFreq / (2 ** (bumpUp + divisor))
    else:
        shifted_freq_ref = refFreq * (2 ** divisor)

    if index == 0:
        lower, upper = shifted_freq_ref * shruti_ratios[0], shifted_freq_ref * shruti_ratios[0]
    elif index == 1:
        lower, upper = shifted_freq_ref * shruti_ratios[1], shifted_freq_ref * shruti_ratios[2]
    elif index == 2:
        lower, upper = shifted_freq_ref * shruti_ratios[3], shifted_freq_ref * shruti_ratios[4]
    elif index == 3:
        lower, upper = shifted_freq_ref * shruti_ratios[5], shifted_freq_ref * shruti_ratios[6]
    elif index == 4:
        lower, upper = shifted_freq_ref * shruti_ratios[7], shifted_freq_ref * shruti_ratios[8]
    elif index == 5:
        lower, upper = shifted_freq_ref * shruti_ratios[9], shifted_freq_ref * shruti_ratios[10]
    elif index == 6:
        lower, upper = shifted_freq_ref * shruti_ratios[11], shifted_freq_ref * shruti_ratios[12]
    elif index == 7:
        lower, upper = shifted_freq_ref * shruti_ratios[13], shifted_freq_ref * shruti_ratios[13]
    elif index == 8:
        lower, upper = shifted_freq_ref * shruti_ratios[14], shifted_freq_ref * shruti_ratios[15]
    elif index == 9:
        lower, upper = shifted_freq_ref * shruti_ratios[16], shifted_freq_ref * shruti_ratios[17]
    elif index == 10:
        lower, upper = shifted_freq_ref * shruti_ratios[18], shifted_freq_ref * shruti_ratios[19]
    elif index == 11:
        lower, upper = shifted_freq_ref * shruti_ratios[20], shifted_freq_ref * shruti_ratios[21]
    

    return (lower * (2 ** (-bandwidth / 1200)), upper * (2 ** (bandwidth / 1200)))  


@njit
def poolMask(pitch, sampleRate, winLength, refPitch=69, refFreq=440):
    """Computes the set of frequency indices that are assigned to a given pitch

    Args:
        pitch (int): MIDI pitch value
        sampleRate (int): Sampling rate
        winLength (int): Window size of Fourier transform
        refPitch (int): Reference pitch (Tonic)(default: 69)
        refFreq (float):  Frequency of reference pitch (Tonic)(default: 440.0)

    Returns:
        im (ndarray): Set of frequency indices
    """
    lower, upper = swaraRange(pitch, 5, refPitch, refFreq)
    index = np.arange(winLength // 2 + 1)
    indexFreq = index * sampleRate / winLength # compute center frequencies for fourier coefficient
    mask = np.logical_and(lower <= indexFreq, indexFreq < upper)
    return index[mask]


@njit
def logFreqSpec(Y, sampleRate, winLength, refPitch, refFreq):
    """Computes a log-frequency spectrogram

    Args:
        Y (ndarray): Magnitude or power spectrogram
        sampleRate (int): Sampling rate
        winLength (int): Window size of Fourier fransform
        refPitch (int): Reference pitch (default: 69)
        refFreq (float): Frequency of reference pitch (default: 440.0)

    Returns:
        Y_LF (ndarray): Log-frequency spectrogram
    """
    Y_LF = np.zeros((128, Y.shape[1]))
    for p in range(128):
        k = poolMask(p, sampleRate, winLength, refPitch, refFreq)
        Y_LF[p, :] = Y[k, :].sum(axis=0)    
    return Y_LF


@njit
def computeSwaragram(stft, sampleRate=22050, winLength=2048, refPitch=69, refFreq=440):
    """Computes a swaragram

    Args:
        stft (ndarray): Magnitude or power spectrogram
        sampleRate (int): Sampling rate
        winLength (int): Window size of Fourier fransform
        refPitch (int): Reference pitch (default: 69)
        refFreq (float): Frequency of reference pitch (default: 440.0)

    Returns:
        C (ndarray): Swaragram matrix
    """
    Y = np.abs(stft) ** 2
    Y_LF = logFreqSpec(Y, sampleRate, winLength, refPitch, refFreq)        
    C = np.zeros((12, Y_LF.shape[1]))
    p = np.arange(128)
    for c in range(12):
        mask = (p % 12) == c
        C[c, :] = Y_LF[mask, :].sum(axis=0)

    return C

def plotSwaragram(swara, t, save=False):
    """Plot swaragram

    Args:
        swara: Swaragram matrix
        t: time indices
        save: save plot or not
    """
    plt.figure(figsize=(15, 5))

    left = min(t)
    right = max(t)
    lower = 0
    upper = 12
 
    chroma_label = ['S', 'r', 'R', 'g', 'G', 'M', 'M`', 'P', 'd', 'D', 'n', 'N']
    plt.imshow(swara, origin='lower', aspect='auto', extent=[t[0], t[-1], 0, 12])
    plt.clim([0, 100])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Swara')
    plt.colorbar()
    plt.yticks(np.arange(12) + 0.5, chroma_label)
    plt.tight_layout()

    if save:
        plt.savefig('swaragram.png')
    else:
        plt.show()