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
import struct
import sympy as sp # Sympy for systems of equations. Used in TDOA function.
from sympy.solvers import solve

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

    vals = struct.unpack("!12I", packet)


    return np.array(vals) # Update it

def phase_difference(input1, input2):
    
    # add linear to circular and use node-fft
    Signal1 = circular2linear(writepointer,input1)
    Signal2 = circular2linear(writepointer,input2)

    F1, D1 = nodeFFT(Signal1, DataRate)
    F2, D2 = nodeFFT(Signal2, DataRate)
   
    # Compute phase spectra
    phase_spectrum1 = np.angle(D1)
    phase_spectrum2 = np.angle(D2)

    # Calculate phase difference spectrum
    phase_diff_spectrum = phase_spectrum1 - phase_spectrum2
    
    for i in range(1,len(phase_diff_spectrum)):
        if phase_diff_spectrum[i] < 0:
            phase_diff_spectrum[i] = phase_diff_spectrum[i] + (2 * np.pi)

    # Convert phase difference to time delay (optional)
    time_delay = phase_diff_spectrum[1:] / (2 * np.pi * F1[1:])

    return(phase_diff_spectrum, time_delay)

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

def TDOA(Mic1, Mic2, Mic3, Mic4):
    SpeedOfSound = 345                                  # Speed of sound at high altitude in m/s
    ####################################################################
    # Include microphone coordinate information from Sam's code here...#
    ####################################################################
    x1,y1,z1,x2,y2,z2,x3,y3,z3 = sp.symbols("x1,y1,z1,x2,y2,z2,x3,y3,z3")   # Make X1-3, Y1-3, and Z1-3 symbols
    phase11, time11 = phase_difference(Mic1, Mic5)  # Find time difference for each signal
    phase12, time12 = phase_difference(Mic1, Mic5)
    phase13, time13 = phase_difference(Mic1, Mic5)
    phase14, time14 = phase_difference(Mic1, Mic5)  ###### Do we need to calculate all this again here? Maybe?
    phase21, time21 = phase_difference(Mic1, Mic5)
    phase22, time22 = phase_difference(Mic1, Mic5)
    phase23, time23 = phase_difference(Mic1, Mic5)
    phase24, time24 = phase_difference(Mic1, Mic5)
    phase31, time31 = phase_difference(Mic1, Mic5)
    phase32, time32 = phase_difference(Mic1, Mic5)
    phase33, time33 = phase_difference(Mic1, Mic5)
    phase34, time34 = phase_difference(Mic1, Mic5)

    timediffArray = [time11,time12,time13,time14,time21,time22,time23,time24,time31,time32,time33,time34]
    minVal = min(timediffArray)                         # Find the lowest time difference
    timediffArray = [x - minVal for x in timediffArray] # Subtract the lowest time difference from each value

    # Using the location information from Sam's code, calculate the Time Difference Of Arrival in 3 sets, then average the answers
    set11 = sp.Eq(np.sqrt((x1 - mic1[0])^2 + (y1 - mic1[1])^2 + (z1 - mic1[2])^2) - np.sqrt((x1 - mic5[0])^2 + (y1 - mic5[1])^2 + (z1 - mic5[2])^2) - (SpeedOfSound*timediffArray[0]))
    set12 = sp.Eq(np.sqrt((x1 - mic5[0])^2 + (y1 - mic5[1])^2 + (z1 - mic5[2])^2) - np.sqrt((x1 - mic7[0])^2 + (y1 - mic7[1])^2 + (z1 - mic7[2])^2) - (SpeedOfSound*timediffArray[1]))
    set13 = sp.Eq(np.sqrt((x1 - mic7[0])^2 + (y1 - mic7[1])^2 + (z1 - mic7[2])^2) - np.sqrt((x1 - mic9[0])^2 + (y1 - mic9[1])^2 + (z1 - mic9[2])^2) - (SpeedOfSound*timediffArray[2]))
    set14 = sp.Eq(np.sqrt((x1 - mic9[0])^2 + (y1 - mic9[1])^2 + (z1 - mic9[2])^2) - np.sqrt((x1 - mic1[0])^2 + (y1 - mic1[1])^2 + (z1 - mic1[2])^2) - (SpeedOfSound*timediffArray[3]))

    set21 = sp.Eq(np.sqrt((x2 - mic6[0])^2 + (y2 - mic6[1])^2 + (z2 - mic6[2])^2) - np.sqrt((x2 - mic8[0])^2 + (y2 - mic8[1])^2 + (z2 - mic8[2])^2) - (SpeedOfSound*timediffArray[4]))
    set22 = sp.Eq(np.sqrt((x2 - mic8[0])^2 + (y2 - mic8[1])^2 + (z2 - mic8[2])^2) - np.sqrt((x2 - mic3[0])^2 + (y2 - mic3[1])^2 + (z2 - mic3[2])^2) - (SpeedOfSound*timediffArray[5]))
    set23 = sp.Eq(np.sqrt((x2 - mic3[0])^2 + (y2 - mic3[1])^2 + (z2 - mic3[2])^2) - np.sqrt((x2 - mic1[0])^2 + (y2 - mic1[1])^2 + (z2 - mic1[2])^2) - (SpeedOfSound*timediffArray[6]))
    set24 = sp.Eq(np.sqrt((x2 - mic1[0])^2 + (y2 - mic1[1])^2 + (z2 - mic1[2])^2) - np.sqrt((x2 - mic6[0])^2 + (y2 - mic6[1])^2 + (z2 - mic6[2])^2) - (SpeedOfSound*timediffArray[7]))

    set31 = sp.Eq(np.sqrt((x3 - mic2[0])^2 + (y3 - mic2[1])^2 + (z3 - mic2[2])^2) - np.sqrt((x3 - mic4[0])^2 + (y3 - mic4[1])^2 + (z3 - mic4[2])^2) - (SpeedOfSound*timediffArray[8]))
    set32 = sp.Eq(np.sqrt((x3 - mic4[0])^2 + (y3 - mic4[1])^2 + (z3 - mic4[2])^2) - np.sqrt((x3 - mic9[0])^2 + (y3 - mic9[1])^2 + (z3 - mic9[2])^2) - (SpeedOfSound*timediffArray[9]))
    set33 = sp.Eq(np.sqrt((x3 - mic9[0])^2 + (y3 - mic9[1])^2 + (z3 - mic9[2])^2) - np.sqrt((x3 - mic6[0])^2 + (y3 - mic6[1])^2 + (z3 - mic6[2])^2) - (SpeedOfSound*timediffArray[10]))
    set34 = sp.Eq(np.sqrt((x3 - mic3[0])^2 + (y3 - mic3[1])^2 + (z3 - mic3[2])^2) - np.sqrt((x3 - mic1[0])^2 + (y3 - mic1[1])^2 + (z3 - mic1[2])^2) - (SpeedOfSound*timediffArray[11]))

    Solution1 = sp.solve([set11,set12,set13,set14],[x1,y1,z1],dict=True)    # Note to future self: check sp.solve to see how it outputs the answers... 
    Solution2 = sp.solve([set21,set22,set23,set24],[x1,y1,z1],dict=True)    # I think this is correct, unless x1 has multiple answers (eg. x^2 -4 = [-2,2])
    Solution3 = sp.solve([set31,set32,set33,set34],[x1,y1,z1],dict=True)

    Xposition = (Solution1[0]+Solution2[0]+Solution3[0])/3
    Yposition = (Solution1[1]+Solution2[1]+Solution3[1])/3
    Zposition = (Solution1[2]+Solution2[2]+Solution3[2])/3

    Xpos = round(Xposition, 4)
    Ypos = round(Yposition, 4)
    Zpos = round(Zposition, 4)

    return(Xpos,Ypos,Zpos)

    
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
