from essentia import *
from essentia.standard import *
from numpy import *
import matplotlib.pyplot as plt
import collections as cll
from collections import Counter
from scipy import signal

file_vibrato = "Neumman-07 render 026.wav"
file_no_vibrato = "Neumman-05 render 008.wav"
file_template = "153765__carlos-vaquero__violoncello-a-3-tenuto-vibrato.wav"

# Obtain the pitch trajectory
hopSize = 128
frameSize = 2048
sampleRate = 44100
guessUnvoiced = True
binResolution = 10

run_windowing = Windowing(type='hann', zeroPadding=4 * frameSize)  # Hann window with x4 zero padding
run_spectrum = Spectrum(size=frameSize * 4)
run_spectral_peaks = SpectralPeaks(minFrequency=1,
                                   maxFrequency=20000,
                                   maxPeaks=100,
                                   sampleRate=sampleRate,
                                   magnitudeThreshold=0,
                                   orderBy="frequency")
run_pitch_salience_function = PitchSalienceFunction()
run_pitch_salience_function_peaks = PitchSalienceFunctionPeaks()
run_pitch_contours = PitchContours(hopSize=hopSize)
run_pitch_contours_melody = PitchContoursMelody(guessUnvoiced=guessUnvoiced,
                                                hopSize=hopSize,binResolution=binResolution)

files = [file_no_vibrato,file_vibrato]
derivative_vector = []
second_derivative_vector = []
pitch_vector = []
n_frames_vector = []

nonzero_indexes_vector = []
differences_vector = []
mean_vector = []
peaknumber = 0
percentage_vector = []
peaknumber_vector = []
length_vector = []
time_vector = []

for file in files:
    audio = MonoLoader(filename=file)()
    audio = EqualLoudness()(audio)

    # ... and create a Pool
    pool = Pool();

    # 2. Cut audio into frames and compute for each frame:
    #    spectrum -> spectral peaks -> pitch salience function -> pitch salience function peaks
    for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        frame = run_windowing(frame)
        spectrum = run_spectrum(frame)
        peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)

        salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
        salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)

        pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
        pool.add('allframes_salience_peaks_saliences', salience_peaks_saliences)

    # 3. Now, as we have gathered the required per-frame data, we can feed it to the contour
    #    tracking and melody detection algorithms:
    contours_bins, contours_saliences, contours_start_times, duration = run_pitch_contours(
        pool['allframes_salience_peaks_bins'],
        pool['allframes_salience_peaks_saliences'])
    pitch, confidence = run_pitch_contours_melody(contours_bins,
                                                  contours_saliences,
                                                  contours_start_times,
                                                  duration)

    # Eliminating the noise at the beginnig and at the end
    # abs_confidence = abs(confidence)
    # m_common = cll.Counter(abs_confidence).most_common()[0]
    # first =[index for index, val in enumerate(abs_confidence) if val == m_common[0]][0]
    # last = [index for index, val in enumerate(abs_confidence) if val == m_common[0]][-1]
    # pitch[0:first+1] = 0
    # pitch[last: ] = 0
    pitch_vector.append(pitch)

    n_frames = len(pitch)
    n_frames_vector.append(n_frames)
    time = arange(0,duration,duration/len(pitch))
    time_vector.append(time)

for i in range(len(pitch_vector)):
    plt.figure(1)
    plt.plot(range(len(pitch_vector[i])),pitch_vector[i],'b')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pitch (Hz)')
    plt.suptitle("Predominant melody pitch")
    # plt.ylim(-20,20)

# First create some templates from a signal with vibrato on the same pitch,
# later they can be normalized
audio_template = MonoLoader(filename=file_template)()
audio_template = EqualLoudness()(audio_template)

# ... and create a Pool
pool = Pool();

# 2. Cut audio into frames and compute for each frame:
#    spectrum -> spectral peaks -> pitch salience function -> pitch salience function peaks
for frame in FrameGenerator(audio_template, frameSize=frameSize, hopSize=hopSize):
    frame = run_windowing(frame)
    spectrum = run_spectrum(frame)
    peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)

    salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
    salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)

    pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
    pool.add('allframes_salience_peaks_saliences', salience_peaks_saliences)

# 3. Now, as we have gathered the required per-frame data, we can feed it to the contour
#    tracking and melody detection algorithms:
contours_bins, contours_saliences, contours_start_times, duration = run_pitch_contours(
    pool['allframes_salience_peaks_bins'],
    pool['allframes_salience_peaks_saliences'])
pitch_template, confidence_template = run_pitch_contours_melody(contours_bins,
                                              contours_saliences,
                                              contours_start_times,
                                              duration)

pitch_template = pitch_template[550:700]

n_frames_template = len(pitch_template)
time_template = arange(0, duration, duration / len(pitch_template))

plt.figure(2)
plt.plot(range(len(pitch_template)),pitch_template,'b')
plt.xlabel('Time (seconds)')
plt.ylabel('Pitch (Hz)')
plt.suptitle("Predominant melody pitch of the template")

# then compare them with the same note's pitch trajectory of different audio
# for both vibrato and no vibrato case

correlate_vib = signal.correlate(pitch_vector[1],pitch_template)
correlate_no_vib = signal.correlate(pitch_vector[0],pitch_template)

pass