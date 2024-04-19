import numpy as np
import matplotlib.pyplot as plt

# Define the length of the white noise signal
length = 1000
sampleRate = 25000

# Generate white noise signal
white_noise = np.random.normal(0, 1, length)

# Take the FFT of the white noise signal
fft_result = np.fft.fft(white_noise)

# Calculate the frequencies corresponding to the FFT result
frequencies = np.fft.fftfreq(length,1/sampleRate)

# Plot the magnitude spectrum of the FFT result
plt.plot(frequencies, np.abs(fft_result))
plt.title('Magnitude Spectrum of White Noise Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()