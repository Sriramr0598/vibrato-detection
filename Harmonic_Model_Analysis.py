from essentia.standard import *
from pylab import *
from numpy import *
from matplotlib.pylab import *
import numpy as np

loader = MonoLoader(filename = 'Neumman-32 render 001.wav')
audio = loader()

frame_vector = []
pitch_vector = []
loudness_vector = []
pitch_confidence_vector = []

time = np.linspace(0,len(audio)/sampleRate,num=len(audio))
frame_to_time = []
local_time = 0

hopSize = 512
frameSize = 1024
sampleRate = 44100

w=Windowing(type='hann')
spectrum=Spectrum()
run_pitchContourSegmentation = PitchContourSegmentation (hopSize=hopSize)
pitch_detect = PitchYinFFT(frameSize=frameSize,sampleRate=sampleRate)
loudness = Loudness()
run_fft = FFT(size=1024)

for frame in FrameGenerator(audio,frameSize,hopSize):
    spec = spectrum(w(frame))
    pitch, pitchConfidence = pitch_detect(spec)
    pitch_vector.append(pitch)
    pitch_confidence_vector.append(pitchConfidence)
    frame_vector.append(frame)


