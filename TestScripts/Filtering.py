from scipy.signal import butter, sosfilt
import numpy as np
import matplotlib.pyplot as plt

DataRate = 25000
def butterworth_coef(cutoff, fs, order=5):
    # Calculate Parameters
    nyq= 0.5*fs
    normal_cutoff = cutoff/nyq
    # Plug parameters into butter to return coefficents for filter
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sos

# Call this function for filtering
def butter_filter(data, cutoff, fs, order=5):
    # Calls function to find the filter coefficents
    sos = butterworth_coef(cutoff, fs, order=order)
    # Uses filter coefs to filter the data
    filtered = sosfilt(sos, data)
    return filtered

def nodeFFT(array):
    sampleRate = 25000
    # Takes the real FFT of the data
    Fdomain = np.fft.rfft(array)
    Frequency = np.fft.rfftfreq(np.size(array),1/sampleRate)

    # Finds the strongest frequencies
    #threshold = .99 * max(abs(Fdomain))
    #print(threshold)
    #mask = abs(Fdomain) > threshold
    #print(mask)
    #Fpeaks = Frequency[mask]
    #Dpeaks = Fdomain[mask]
    #print(Fpeaks)
    #return(Fpeaks,Dpeaks)
    return(Fdomain, Frequency)

length = 10000
white_noise = np.random.normal(0, 1, length)
FilteredData = butter_filter(white_noise,3000,25000)
F, D = nodeFFT(FilteredData)
plt.plot(D,np.abs(F))
plt.title('White Noise Signal')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()



