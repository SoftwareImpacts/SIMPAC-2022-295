Documentation of 'core' module

NAME
    swaragram.core

FUNCTIONS

    computeSwaragram(stft, sampleRate=22050, winLength=2048, refPitch=69, refFreq=440)
        Computes a swaragram
        
        Args:
            stft (ndarray): Magnitude or power spectrogram
            sampleRate (int): Sampling rate
            winLength (int): Window size of Fourier fransform
            refPitch (int): Reference pitch (default: 69)
            refFreq (float): Frequency of reference pitch (default: 440.0)
        
        Returns:
            C (ndarray): Swaragram matrix
    
    logFreqSpec(Y, sampleRate, winLength, refPitch, refFreq)
        Computes a log-frequency spectrogram
        
        Args:
            Y (ndarray): Magnitude or power spectrogram
            sampleRate (int): Sampling rate
            winLength (int): Window size of Fourier fransform
            refPitch (int): Reference pitch (default: 69)
            refFreq (float): Frequency of reference pitch (default: 440.0)
        
        Returns:
            Y_LF (ndarray): Log-frequency spectrogram
    
    plotSwaragram(swara, t, save=False)
        Plot swaragram
        
        Args:
            swara: Swaragram matrix
            t: time indices
            save: save plot or not
    
    poolMask(pitch, sampleRate, winLength, refPitch=69, refFreq=440)
        Return a mask for aggregating frequecy values
        
        Args:
            pitch (int): MIDI pitch value
            sampleRate (int): Sampling rate
            winLength (int): Window size of Fourier transform
            refPitch (int): Reference pitch (Tonic)(default: 69)
            refFreq (float):  Frequency of reference pitch (Tonic)(default: 440.0)
        
        Returns:
            im (ndarray): Set of frequency indices
    
    swaraRange(pitch, bandwidth=0, refPitch=69, refFreq=440)
        Return range for a pitch value based on Shruti
        
        
        Args:
            pitch (int): MIDI pitch value
            bandwidth (int): Tolerance for the upper and lower frequency values of a given pitch 
            refPitch (int): Reference pitch (Tonic) (default: 69)
            refFreq (float): Frequency of reference pitch (Tonic) (default: 440.0)
        
        Returns:
            lower (float): Lower value of the pitch frequency adjusted with bandwidth
            upper (float): Upper value of the pitch frequency adjusted with bandwidth

