Documentation of 'utils' module

NAME
    swaragram.utils

FUNCTIONS

    computeRefPitch(tonic)
        Compute tonic pitch and its frequency
        
        Args:
            tonic (float): Tonic of the audio sample
        
        Returns:
            tonicMidiPitch (int): Nearest midi pitch to tonic
            midiPitchFreq (float): Corresponding frequency to nearest midi pitch
    
    computeSTFT(x, win_length=2048, hop_length=512, window='hann', pad_mode='constant')
        Calculate STFT of an audio signal
        
        Args:
            x (np.array): Audio sample values sampled at sampleRate
            win_length (int): Window size of Fourier fransform (default=2048)
            hop_length (int): Hop size of Fourier fransform (default=512)
            window  (str): Window function (default='hann')
            pad_mode (str): Padding mode used in windowing (default='constant' -> fill with zeros)
        Returns:
            stft: STFT magnitude matrix
    
    getTonic(x)
        Get the tonic of the audio signal
        
        Args:
            x (np.array): Audio signal vector (numpy array)
        Returns:
            im (float): Tonic of the audio signal
    
    loadAudio(filePath, sampleRate=22050)
        Load audio sample
        
        Args:
            filePath (str): Path of the audio file
            sampleRate (int): Sampling rate
        Returns:
            x (np.ndarray): Audio sample values sampled at sampleRate
            sr (int): Sampling rate
    
    logCompression(x)
        Log compression the input value
        
        Args:
            x : Magnitude/ numpy array / numpy matrix
        Returns:
            im : Log compression of input

