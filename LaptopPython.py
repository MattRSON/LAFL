# Written my Jai Bhullar and Mathew Rawson

import socket # For network connection
import threading # For threading used for plot
import matplotlib.pyplot as plt # For the plot
from matplotlib.animation import FuncAnimation # For continuous updates to the plot
from collections import deque # Data structure. Allows us to add and remove from both sides of the array
import signal # Used to safe shutdown the script
import sys # Also used to safe shutdown the script
import numpy as np # For extra number manipulation
from timeit import default_timer as timer
import time
#from sklearn.preprocessing import StandardScaler

## Setting up the network with the name of computer and what port its sending data on
#HOST = "LAFL"   # Hostname
HOST = "127.0.0.1" # Loopback for HardwareEmulator.py
PORT = 65432    # Port

MAX_DATA_POINTS = 10000
GRAPHED_DATA_POINTS = 64

DataRate = 10000 #Hz

# Global Data Lock
data_lock = threading.Lock() # Prevents both threads from trying to modify a variable at the same time
data_value = np.zeros((12,MAX_DATA_POINTS)) # Initializes the global data variable. This is the data from the ADCs
value = np.zeros((12,MAX_DATA_POINTS))
writepointer = 0

def phase_difference():
    
    with data_lock: # If the thread has control of the variable
        value = data_value # Grab the most recent update

    # add linear to circular and use node-fft


    # Perform Fourier transforms
    fft_signal1 = np.fft.rfft(value[0,:])
    fft_signal2 = np.fft.rfft(value[1,:])
    #print(fft_signal1)
    #print(fft_signal2)
    # Compute phase spectra
    phase_spectrum1 = np.angle(fft_signal1)
    phase_spectrum2 = np.angle(fft_signal2)
    
    # Test mask for the signals
    threshold_spectrum1 = 0.5 * max(abs(fft_signal1))
    mask1 = fft_signal1 > threshold_spectrum1
    peaks1 = phase_spectrum1[mask1]
    print(peaks1)
    threshold_spectrum2 = 0.5 * max(abs(fft_signal2))
    mask2 = fft_signal2 > threshold_spectrum2
    peaks2 = phase_spectrum2[mask2]
    print(peaks2)

    # Calculate phase difference spectrum
    phase_diff_spectrum = phase_spectrum1 - phase_spectrum2
    
    # Convert phase difference to time delay (optional)
    #freq = np.fft.fftfreq(len(value[0,:]), 1/DataRate)
    #time_delay = phase_diff_spectrum / (2 * np.pi * freq)

    #print(phase_spectrum1)
    #print(phase_spectrum2)
    #print(phase_diff_spectrum)

    #print(time_delay)

# Thread to receive data from PI (No Delay)
def nodeA(pointer):
    
    start = timer()
    packet = s.recv(48)
    bigint = int.from_bytes(packet,"little")
    value[0,pointer] = bigint & 0xffffffff
    value[1,pointer] = (bigint >> 32) & 0xffffffff
    value[2,pointer] = (bigint >> 64) & 0xffffffff
    value[3,pointer] = (bigint >> 96) & 0xffffffff
    value[4,pointer] = (bigint >> 128) & 0xffffffff
    value[5,pointer] = (bigint >> 160) & 0xffffffff
    value[6,pointer] = (bigint >> 192) & 0xffffffff
    value[7,pointer] = (bigint >> 224) & 0xffffffff
    value[8,pointer] = (bigint >> 256) & 0xffffffff
    value[9,pointer] = (bigint >> 288) & 0xffffffff
    value[10,pointer] = (bigint >> 320) & 0xffffffff
    value[11,pointer] = (bigint >> 352) & 0xffffffff

    pointer = (pointer + 1) % MAX_DATA_POINTS
    end = timer()

    if (end-start) < (1/DataRate):
        time.sleep((1/DataRate)-(end-start))
    #else:
    #    print("oh no!") # If code is not keeping up we have a problem

    return(value, pointer) # Update it

def nodeFFT():

    with data_lock: # If the thread has control of the variable
        value = data_value # Grab the most recent update
    # Sets up a temp list that takes dat from the circular buffer and puts it in order

    mic1 = circular2linear(writepointer, value[0,:])
    

    # Takes the real FFT of the data
    Fdomain = abs(np.fft.rfft(mic1))
    Frequency = np.fft.rfftfreq(mic1.size,1/DataRate)

    # Finds the strongest frequencies
    threshold = 0.5 * max(abs(Fdomain))
    mask = Fdomain > threshold
    peaks = Frequency[mask]
    print(peaks)

    # Sets the thread to run again in 10 seconds
    FFT_Thread = threading.Timer(5,nodeFFT)
    FFT_Thread.daemon = True
    FFT_Thread.start()

# rearanges a circular buffer into a linear one give the new 2 old pointer
def circular2linear(pointer, array):
    size = np.size(array)
    tempValue = np.zeros(size)
    tempValue[0:pointer-1] = np.flipud(value[0,0:pointer-1])
    tempValue[pointer:size] = np.flipud(value[0,pointer:size])
    return tempValue

# Function to shutdown script safely
def signal_handler(sig, frame):
    print("ABORTING")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

FFT_Thread = threading.Timer(5,nodeFFT)
FFT_Thread.daemon = True
FFT_Thread.start()


#PhaseDiff_Thread2 = threading.Timer(5,phase_difference)
#PhaseDiff_Thread2.daemon = True
#PhaseDiff_Thread2.start()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # Checks to see if the Rpi server is running
        s.connect((HOST, PORT)) # Tries to connect to the server   
        while True:
            data_value, writepointer = nodeA(writepointer)
