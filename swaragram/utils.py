import numpy as np
from numba import njit
import librosa
import essentia
import essentia.standard


# Utility functions

def loadAudio(filePath, sampleRate=22050):
    """ Load audio sample

    Args:
        filePath (str): Path of the audio file
        sampleRate (int): Sampling rate
    Returns:
        x (np.ndarray): Audio sample values sampled at sampleRate
        sr (int): Sampling rate
    """
    x , sr = librosa.load(path=filePath, sr=sampleRate)
    #x = essentia.standard.MonoLoader(filename=filePath, sampleRate=sampleRate)()
    return x, sr


def computeSTFT(x, win_length=2048, hop_length=512, window='hann', pad_mode='constant'):
    """ Calculate STFT of an audio signal

    Args:
        x (np.array): Audio sample values sampled at sampleRate
        win_length (int): Window size of Fourier fransform (default=2048)
        hop_length (int): Hop size of Fourier fransform (default=512)
        window  (str): Window function (default='hann')
        pad_mode (str): Padding mode used in windowing (default='constant' -> fill with zeros)
    Returns:
        stft: STFT magnitude matrix
    """
    stft = librosa.stft(x, n_fft=win_length, hop_length=hop_length, win_length=win_length, window=window, pad_mode=pad_mode)
    return stft


def computeRefPitch(tonic):
    """ Compute tonic pitch and its frequency

    Args:
        tonic (float): Tonic of the audio sample

    Returns:
        tonicMidiPitch (int): Nearest midi pitch to tonic
        midiPitchFreq (float): Corresponding frequency to nearest midi pitch 

    """
    freqGen = np.vectorize(lambda x: 440 * 2 ** ((x - 69) / 12))
    midiPitchFreq = freqGen(np.arange(0,128))

    tonicMidiPitch = np.abs(tonic - midiPitchFreq).argmin()
    return tonicMidiPitch, midiPitchFreq[tonicMidiPitch]


@njit
def logCompression(x):
    """ Log compression the input value
    
    Args:
        x : Magnitude/ numpy array / numpy matrix
    Returns:
        im : Log compression of input
    """
    eps = 2.220446049250313e-16 # for turning jit on
    return (23 * np.log10(eps + x))


def getTonic(x):
    """ Get the tonic of the audio signal

    Args:
        x (np.array): Audio signal vector (numpy array)
    Returns:
        im (float): Tonic of the audio signal

    """
    return essentia.standard.TonicIndianArtMusic().compute(x)

