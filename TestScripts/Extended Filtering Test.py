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
import time
import struct
from scipy.signal import butter, sosfilt # Functions for applying a lowpass butterworth
import sympy as sp # Sympy for systems of equations. Used in TDOA function.
from sympy.solvers import solve


## Setting up the network with the name of computer and what port its sending data on
#HOST = "LAFL"   # Hostname
HOST = "127.0.0.1" # Loopback for HardwareEmulator.py
PORT = 65432    # Port

MAX_DATA_POINTS = 10000

DataRate = 7400 #Hz

# Filter Parameters
low_cutoff = 100 #Hz
high_cutoff = 3000 #Hz
order = 6

# Global Data Lock
data_lock = threading.Lock() # Prevents both threads from trying to modify a variable at the same time
data_value = np.zeros((12,MAX_DATA_POINTS)) # Initializes the global data variable. This is the data from the ADCs
value = np.zeros((12,MAX_DATA_POINTS))
writepointer = 0

# Thread to receive data from PI (No Delay)
def phase_difference(input1, input2):
    

    F1, D1 = nodeFFT(input1, DataRate)
    F2, D2 = nodeFFT(input2, DataRate)
   
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

    # Takes the real FFT of the data
    Fdomain = np.fft.rfft(array)
    Frequency = np.fft.rfftfreq(np.size(array),1/sampleRate)

    return(Fdomain, Frequency)


# rearranges a circular buffer into a linear one give the new 2 old pointer
def circular2linear(index, array):
    size = np.size(array)
    tempValue = np.zeros(size)
    tempValue = np.hstack((array[index:], array[:index]))
    return tempValue

def TDOA(Data, pointer, Inputs):
    # add linear to circular and use node-fft
    FixedInput1 = circular2linear(pointer,Data[Inputs[0]-1])
    FixedInput2 = circular2linear(pointer,Data[Inputs[1]-1])
    FixedInput3 = circular2linear(pointer,Data[Inputs[2]-1])
    FixedInput4 = circular2linear(pointer,Data[Inputs[3]-1])

    filteredInput1 = butter_filter(FixedInput1, 3000, 25000, 6)
    filteredInput2 = butter_filter(FixedInput2, 3000, 25000, 6)
    filteredInput3 = butter_filter(FixedInput3, 3000, 25000, 6)
    filteredInput4 = butter_filter(FixedInput4, 3000, 25000, 6)

    SpeedOfSound = 345                                  # Speed of sound at high altitude in m/s

    x1,y1,z1 = sp.symbols("x1,y1,z1")   # Make X1-3, Y1-3, and Z1-3 symbols
    phase11, time11 = phase_difference(filteredInput1, filteredInput2)  # Find time difference for each signal
    phase12, time12 = phase_difference(filteredInput2, filteredInput3)
    phase13, time13 = phase_difference(filteredInput3, filteredInput4)  #### This needs to match case with the set equations
    phase14, time14 = phase_difference(filteredInput4, filteredInput1)  ###### Do we need to calculate all this again here? Maybe?

    Locations = np.zeros((4,3))
    for x in range(4):
        Locations[x] = array_place(Inputs[x])

    timediffArray = [time11,time12,time13,time14]#,time21,time22,time23,time24,time31,time32,time33,time34]
    minVal = np.min(timediffArray)                         # Find the lowest time difference
    relativeTime = [x - minVal for x in timediffArray] # Subtract the lowest time difference from each value


    # Using the location information from Sam's code, calculate the Time Difference Of Arrival in 3 sets, then average the answers
    set11 = sp.Eq(np.sqrt((x1 - Locations[0][0])^2 + (y1 - Locations[0][1])^2 + (z1 - Locations[0][2])^2) - np.sqrt((x1 - Locations[1][0])^2 + (y1 - Locations[1][1])^2 + (z1 - Locations[1][2])^2) - (SpeedOfSound*(relativeTime[0]-relativeTime[1])))
    set12 = sp.Eq(np.sqrt((x1 - Locations[1][0])^2 + (y1 - Locations[1][1])^2 + (z1 - Locations[1][2])^2) - np.sqrt((x1 - Locations[2][0])^2 + (y1 - Locations[2][1])^2 + (z1 - Locations[2][2])^2) - (SpeedOfSound*(relativeTime[1]-relativeTime[2])))
    set13 = sp.Eq(np.sqrt((x1 - Locations[2][0])^2 + (y1 - Locations[2][1])^2 + (z1 - Locations[2][2])^2) - np.sqrt((x1 - Locations[3][0])^2 + (y1 - Locations[3][1])^2 + (z1 - Locations[3][2])^2) - (SpeedOfSound*(relativeTime[2]-relativeTime[3])))
    set14 = sp.Eq(np.sqrt((x1 - Locations[3][0])^2 + (y1 - Locations[3][1])^2 + (z1 - Locations[3][2])^2) - np.sqrt((x1 - Locations[0][0])^2 + (y1 - Locations[0][1])^2 + (z1 - Locations[0][2])^2) - (SpeedOfSound*(relativeTime[3]-relativeTime[0])))

    Solution1 = sp.solve([set11,set12,set13,set14],[x1,y1,z1],dict=True)    # Note to future self: check sp.solve to see how it outputs the answers... 
    print(Solution1)

    
# Function to shutdown script safely
def signal_handler(sig, frame):
    print("ABORTING")
    sys.exit(0)

# Function for Butterworth lowpass filtering
def butterworth_coef(fs, order=5):
    # Calculate Parameters
    nyq= 0.5*10000
    low = 100/nyq
    high = 4000/nyq
    # Plug parameters into butter to return coefficents for filter
    sos = butter(order, [low, high], btype='band', analog=False, output='sos')
    print(sos)
    return sos

# Call this function for filtering
def butter_filter(data):
    # Uses filter coefs to filter the data
    sos = [[ 8.52734684e-05,  1.70546937e-04,  8.52734684e-05,  1.00000000e+00, -1.14440525e+00,  7.07260746e-01],
        [ 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,  1.00000000e+00, -1.02151851e+00,  7.39321383e-01],
        [ 1.00000000e+00,  0.00000000e+00, -1.00000000e+00,  1.00000000e+00, -1.31481158e+00,  7.78286489e-01],
        [ 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,  1.00000000e+00, -1.00445753e+00,  8.88486200e-01],
        [ 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,  1.00000000e+00, -1.48224542e+00,  9.15406042e-01]]
    filtered = sosfilt(sos, data)
    
    return filtered


# Function for Microphone placements
def array_place(input):
    # Find midpoint of array
    zed = 1
    array_size = 50
    spacing = 0.125
    middle = array_size/2
    if input == 1:
        positions = np.array([middle-((spacing*4)*np.sin(1.0472)),middle-((spacing*4)*np.cos(1.0472)),zed])
    elif input == 2:
        positions = np.array([middle-((spacing*3)*np.sin(1.0472)),middle-((spacing*3)*np.cos(1.0472)),zed])
    elif input == 3:
        positions = np.array([middle-((spacing*2)*np.sin(1.0472)),middle-((spacing*2)*np.cos(1.0472)),zed])
    elif input == 4:
        positions = np.array([middle-((spacing)*np.sin(1.0472)),middle-((spacing)*np.cos(1.0472)),zed])
    elif input == 5:
        positions = np.array([middle,middle+(4*spacing),zed])
    elif input == 6:
        positions = np.array([middle,middle+(3*spacing),zed])
    elif input == 7:
        positions = np.array([middle,middle+(2*spacing),zed])
    elif input == 8:
        positions = np.array([middle,middle+spacing,zed])
    elif input == 9:
        positions = np.array([middle+((spacing)*np.sin(1.0472)),middle-((spacing)*np.cos(1.0472)),zed])
    elif input == 10:
        positions = np.array([middle+((spacing*2)*np.sin(1.0472)),middle-((spacing*2)*np.cos(1.0472)),zed])
    elif input == 11:
        positions = np.array([middle+((spacing*3)*np.sin(1.0472)),middle-((spacing*3)*np.cos(1.0472)),zed])
    elif input == 12:
        positions = np.array([middle+((spacing*4)*np.sin(1.0472)),middle-((spacing*4)*np.cos(1.0472)),zed])


    return positions

signal.signal(signal.SIGINT, signal_handler)


butterworth_coef(7400,5)
realData = np.fromfile('../Recorded Data/240417_140549.bin', dtype=np.float64)
realData = np.reshape(realData,(12,-1))
points = np.size(realData)/12

filteredInput1 = butter_filter(realData[11,:])

F,D = nodeFFT(filteredInput1,7400)

MaxD1 = np.max(abs(F))
IndmaxD1 = np.where(abs(F) == MaxD1)[0][0]
print(MaxD1)
print(IndmaxD1)
#plt.plot(range(int(points)), filteredInput1)
plt.plot(D,abs(F))
plt.title('Signal')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
               
