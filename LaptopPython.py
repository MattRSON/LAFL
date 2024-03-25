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

## Setting up the network with the name of computer and what port its sending data on
#HOST = "LAFL"   # Hostname
HOST = "127.0.0.1" # Loopback for HardwareEmulator.py
PORT = 65432    # Port

MAX_DATA_POINTS = 10000
GRAPHED_DATA_POINTS = 64

DataRate = 50000 #Hz

# Global Data Lock
data_lock = threading.Lock() # Prevents both threads from trying to modify a variable at the same time
data_value = np.zeros((12,MAX_DATA_POINTS)) # Initializes the global data variable. This is the data from the ADCs
value = np.zeros((12,MAX_DATA_POINTS))
writepointer = 0


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

    if (end-start) < (1/50000):
        time.sleep((1/DataRate)-(end-start))

    return(value, pointer) # Update it

def nodeFFT():

    with data_lock: # If the thread has control of the variable
        value = data_value # Grab the most recent update

    tempValue = np.zeros(MAX_DATA_POINTS)
    tempValue[0:writepointer-1] = np.flipud(value[0,0:writepointer-1])
    tempValue[writepointer:MAX_DATA_POINTS] = np.flipud(value[0,writepointer:MAX_DATA_POINTS])

    Fdomain = abs(np.fft.rfft(tempValue))
    Frequency = np.fft.rfftfreq(tempValue.size,2e-5)

    threshold = 0.5 * max(abs(Fdomain))
    mask = Fdomain > threshold

    peaks = Frequency[mask]
    print(peaks)

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
            data_value, writepointer = nodeA(writepointer)
