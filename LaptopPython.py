# Written by Mathew Rawson, Samuel Myers, Tyler LeMoine, Caise Taylor, and Robert Volkmann
# With assistance from Jai Bhullar of the Cyber Security and Computer Science department

import socket # For network connection
import threading # For threading used for plot
import matplotlib.pyplot as plt # For the plot
from matplotlib.animation import FuncAnimation # For continuous updates to the plot
from collections import deque # Data structure. Allows us to add and remove from both sides of the array
import signal # Used to safe shutdown the script
import sys # Also used to safe shutdown the script
import numpy as np # For extra number manipulation
from timeit import default_timer as timer

## Setting up the network with the name of computer and what port its sending data on
#HOST = "LAFL"   # Hostname
HOST = "127.0.0.1" # Loopback for HardwareEmulator.py
PORT = 65432    # Port

MAX_DATA_POINTS = 10000

DataRate = 10000 #Hz

# Global Data Lock
data_lock = threading.Lock() # Prevents both threads from trying to modify a variable at the same time
data_value = np.zeros((12,MAX_DATA_POINTS)) # Initializes the global data variable. This is the data from the ADCs
value = np.zeros((12,MAX_DATA_POINTS))
writepointer = 0

# Thread to receive data from PI (No Delay)
def nodeA():
    
    packet = b''

    while len(packet) == 0:
        packet = s.recv(48)
    start = timer()

    vals = struct.unpack("!12I", packet)

    end = timer()

    if (end-start) < (1/DataRate):
        time.sleep((1/DataRate)-(end-start))
    else:
        print("oh no!") # If code is not keeping up we have a problem
        print(end-start)

    return np.array(vals) # Update it

def phase_difference():
    
    with data_lock: # If the thread has control of the variable
        value = data_value # Grab the most recent update

    # add linear to circular and use node-fft
    Signal1 = circular2linear(writepointer,value[0,:])
    Signal2 = circular2linear(writepointer,value[1,:])

    F1, D1 = nodeFFT(Signal1, DataRate)
    F2, D2 = nodeFFT(Signal2, DataRate)
   
    # Compute phase spectra
    phase_spectrum1 = np.angle(D1)
    phase_spectrum2 = np.angle(D2)

    # Calculate phase difference spectrum
    phase_diff_spectrum = phase_spectrum1 - phase_spectrum2
    
    # Convert phase difference to time delay (optional)
    #time_delay = phase_diff_spectrum / (2 * np.pi * F1)

    #print(phase_spectrum1)
    #print(phase_spectrum2)
    print(phase_diff_spectrum)

    #print(time_delay)

    PhaseDiff_Thread2 = threading.Timer(5,phase_difference)
    PhaseDiff_Thread2.daemon = True
    PhaseDiff_Thread2.start()

def nodeFFT(array,sampleRate):

    # Sets up a temp list that takes dat from the circular buffer and puts it in order

    # Takes the real FFT of the data
    Fdomain = np.fft.rfft(array)
    Frequency = np.fft.rfftfreq(np.size(array),1/sampleRate)

    # Finds the strongest frequencies
    threshold = 0.5 * max(abs(Fdomain))
    mask = abs(Fdomain) > threshold
    Fpeaks = Frequency[mask]
    Dpeaks = Fdomain[mask]

    return(Fpeaks,Dpeaks)


# rearranges a circular buffer into a linear one give the new 2 old pointer
def circular2linear(index, array):
    size = np.size(array)
    tempValue = np.zeros(size)
    tempValue = np.hstack((array[index:], array[:index]))
    return tempValue

# Function to shutdown script safely
def signal_handler(sig, frame):
    print("ABORTING")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


PhaseDiff_Thread2 = threading.Timer(5,phase_difference)
PhaseDiff_Thread2.daemon = True
PhaseDiff_Thread2.start()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # Checks to see if the Rpi server is running
        s.connect((HOST, PORT)) # Tries to connect to the server   
        while True:
            tmp = nodeA()
            with data_lock: # If the thread has control of the variables
                # Filtering goes here (likely?)               
                data_value[:, writepointer] = tmp
                writepointer = (writepointer + 1) % MAX_DATA_POINTS
