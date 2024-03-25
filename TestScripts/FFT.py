import numpy as np

x = np.arange(0,.001,2e-5)
signal = np.sin(2*3.14*1000*x)
Fdomain = abs(np.fft.fft(signal))
Frequency = np.fft.fftfreq(Fdomain.size,2e-5)
threshold = 0.5 * max(abs(Fdomain))

print(threshold)
mask = abs(Fdomain) > threshold
print(mask)
peaks = Frequency[mask]
print(peaks)