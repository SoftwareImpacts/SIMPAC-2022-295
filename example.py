from swaragram import core, utils
import librosa
import numpy as np

# load audio file
#x , sr = utils.loadAudio('data/Raga_Bhopali.mp3')
x, sr = utils.loadAudio('sample.wav')

# calculate tonic of the indian classical music sample
tonic = utils.getTonic(x)

# calculate STFT with win_length=2048 and hop_length=512
stft = utils.computeSTFT(x)

# compute reference pitch for calculating swaragram
refPitch, refFreq = utils.computeRefPitch(tonic)

# calculate swaragram for a given ICM sample
swara = core.computeSwaragram(stft, refPitch=refPitch, refFreq=refFreq)

# Get time values for plotting swaragram
t = librosa.frames_to_time(np.arange(stft.shape[1]), n_fft=2048)

# Log compress swaragram
swara = utils.logCompression(swara)

# Plot swaragram
core.plotSwaragram(swara, t)
