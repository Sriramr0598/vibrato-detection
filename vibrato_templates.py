import numpy as np
import matplotlib.pyplot as plt

Fs = 20.
l = 0.5
f = np.arange(5., 8.5, 0.25)
e = np.arange(50., 210., 10) # extent in cents
sample = Fs*l
x = np.arange(sample)
# y = np.sin(2 * np.pi * f * x / Fs)
time = np.arange(0., l, 1/Fs)
# amp = pow(2,(cent/1200.))

templates = []

# Peak to peak amplitude is 100 cents, plotted in Hz

for cent in e:
    for freq in f:
        y = np.sin(2 * np.pi * freq * x / Fs)
        amp = pow(2, (cent / 1200.))
        template = y*amp
        # plt.figure()
        # plt.plot(time, y * amp)
        # plt.xlabel('Time (seconds)')
        # plt.ylabel('Frequency (Hz)')
        # plt.title('Vibrato Template')
        # plt.ylim(-20, 20)
        # plt.xlim(0,4)
        # plt.show()
        templates.append(template)

# print d.items()
pass