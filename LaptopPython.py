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
import struct
from scipy.signal import butter, sosfilt # Functions for applying a lowpass butterworth

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

def nodeFFT():

    with data_lock: # If the thread has control of the variables
        value = data_value # Grab the most recent update

        # Sets up a temp list that takes dat from the circular buffer and puts it in order
        tempValue = np.zeros(MAX_DATA_POINTS)
        tempValue[0:writepointer-1] = np.flipud(value[0,0:writepointer-1])
        tempValue[writepointer:MAX_DATA_POINTS] = np.flipud(value[0,writepointer:MAX_DATA_POINTS])

    # Takes the real FFT of the data
    Fdomain = abs(np.fft.rfft(tempValue))
    Frequency = np.fft.rfftfreq(tempValue.size,2e-5)

    # Finds the strongest frequencies
    threshold = 0.5 * max(abs(Fdomain))
    mask = Fdomain > threshold
    peaks = Frequency[mask]
    print(peaks)

    # Sets the thread to run again in 10 seconds
    FFT_Thread = threading.Timer(10,nodeFFT)
    FFT_Thread.daemon = True
    FFT_Thread.start()

# Function to shutdown script safely
def signal_handler(sig, frame):
    print("ABORTING")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

FFT_Thread = threading.Timer(10,nodeFFT)
FFT_Thread.daemon = True
FFT_Thread.start()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # Checks to see if the Rpi server is running
        s.connect((HOST, PORT)) # Tries to connect to the server   
        while True:
            tmp = nodeA()
            with data_lock: # If the thread has control of the variables
                data_value[:, writepointer] = tmp
                writepointer = (writepointer + 1) % MAX_DATA_POINTS