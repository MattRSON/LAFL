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
#import time
import struct
from scipy.signal import butter, sosfilt # Functions for applying a lowpass butterworth
import sympy as sp # Sympy for systems of equations. Used in TDOA function.
#from sympy.solvers import solve
#from sympy.abc import x,y,z
from scipy.optimize import minimize


## Setting up the network with the name of computer and what port its sending data on
#HOST = "LAFL"   # Hostname
HOST = "127.0.0.1" # Loopback for HardwareEmulator.py
PORT = 65432    # Port

MAX_DATA_POINTS = 10000

DataRate = 7400 #Hz

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Global Data Lock
data_lock = threading.Lock() # Prevents both threads from trying to modify a variable at the same time
data_value = np.zeros((12,MAX_DATA_POINTS)) # Initializes the global data variable. This is the data from the ADCs
#value = np.zeros((12,MAX_DATA_POINTS))
writepointer = 0

# Thread to receive data from PI (No Delay)
def nodeA():
    global data_value
    global writepointer
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # Checks to see if the Rpi server is running
        s.connect((HOST, PORT)) # Tries to connect to the server   
        while True:
            packet = b''

            while len(packet) == 0:
                packet = s.recv(48)

            vals = struct.unpack("!12I", packet)

            with data_lock: # If the thread has control of the variables             
                data_value[:, writepointer] = np.array(vals)
                writepointer = (writepointer + 1) % MAX_DATA_POINTS

def phase_difference(input1, input2):
    

    F1, D1 = nodeFFT(input1, DataRate)
    F2, D2 = nodeFFT(input2, DataRate)
    
    # grab only the most prominent frequency 
    MaxD1 = np.max(abs(D1))
    IndmaxD1 = np.where(abs(D1) == MaxD1)[0][0]

    # Compute phase spectra
    phase_spectrum1 = np.angle(D1[IndmaxD1])
    phase_spectrum2 = np.angle(D2[IndmaxD1])

    # Calculate phase difference spectrum
    phase_diff_spectrum = phase_spectrum1 - phase_spectrum2
    
    
    if phase_diff_spectrum < 0:
        phase_diff_spectrum = phase_diff_spectrum + (2 * np.pi)

    # Convert phase difference to time delay (optional)
    if F1[IndmaxD1] > 0:
        time_delay = phase_diff_spectrum / (2 * np.pi * F1[IndmaxD1])
        print(F1[IndmaxD1], F2[IndmaxD1])
    else:
        time_delay = 0

    if F1[IndmaxD1] > 1300 or F1[IndmaxD1] < 700:
        plt.plot(F1,np.abs(D1))
        plt.plot(F2,np.abs(D2))
        plt.show()

    return(phase_diff_spectrum, time_delay)

def nodeFFT(array,sampleRate):

    # Takes the real FFT of the data
    Fdomain = np.fft.fft(array)
    Frequency = np.fft.fftfreq(np.size(array),1/sampleRate)
    return(Frequency,Fdomain)


# rearranges a circular buffer into a linear one give the new 2 old pointer
def circular2linear(index, array):
    size = np.size(array)
    tempValue = np.zeros(size)
    tempValue = np.hstack((array[index:], array[:index]))
    return tempValue

def equation(coords, Locations, relativeTime, SpeedOfSound):
    x1, y1, z1 = coords
    term1 = ((x1 - Locations[0][0])**2 + (y1 - Locations[0][1])**2 + (z1 - Locations[0][2])**2)**0.5
    term2 = ((x1 - Locations[1][0])**2 + (y1 - Locations[1][1])**2 + (z1 - Locations[1][2])**2)**0.5

    return term1 - term2 - SpeedOfSound * (relativeTime[0] - relativeTime[1])

def TDOA(Data, pointer, Inputs):
    # add linear to circular and use node-fft
    FixedInput1 = circular2linear(pointer,Data[Inputs[0]-1])
    FixedInput2 = circular2linear(pointer,Data[Inputs[1]-1])
    FixedInput3 = circular2linear(pointer,Data[Inputs[2]-1])
    FixedInput4 = circular2linear(pointer,Data[Inputs[3]-1])

    filteredInput1 = butter_filter(FixedInput1)
    filteredInput2 = butter_filter(FixedInput2)
    filteredInput3 = butter_filter(FixedInput3)
    filteredInput4 = butter_filter(FixedInput4)

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
    #print(timediffArray)
    minVal = np.min(timediffArray)                         # Find the lowest time difference
    relativeTime = [x - minVal for x in timediffArray] # Subtract the lowest time difference from each value
    
    initial_guess = [0, 0, 0]
    result = minimize(equation, initial_guess, args=(Locations, relativeTime, SpeedOfSound))
    x1_opt, y1_opt, z1_opt = result.x

    return(x1_opt, y1_opt, z1_opt)

    
# Function to shutdown script safely
def signal_handler(sig, frame):
    print("ABORTING")
    sys.exit(0)

def butterworth_coef(cutoff, fs, order=5):
    # Calculate Parameters
    nyq= 0.5*fs
    normal_cutoff = cutoff/nyq
    # Plug parameters into butter to return coefficents for filter
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sos

# Call this function for filtering
def butter_filter(data):
    # Calls function to find the filter coefficents
    sos = [[ 0.32426286,  0.64852571,  0.32426286,  1.,          1.13133225,  0.37447098],
        [ 1.,          2.,          1.,          1.,          1.41580199,  0.71039477],
        [ 1.,          0.,         -1.,          1.,         -0.3949446,  -0.47784437],
        [ 1.,         -2.,          1.,          1.,         -1.86371872,  0.87069052],
        [ 1.,         -2.,          1.,          1.,         -1.94291867,  0.94997688]]
    # Uses filter coefs to filter the data
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

    
    #positions = np.array([[middle-((spacing*4)*np.sin(1.0472)),middle-((spacing*4)*np.cos(1.0472)),zed],[middle-((spacing*3)*np.sin(1.0472)),middle-((spacing*3)*np.cos(1.0472)),zed],[middle-((spacing*2)*np.sin(1.0472)),middle-((spacing*2)*np.cos(1.0472)),zed],[middle-((spacing)*np.sin(1.0472)),middle-((spacing)*np.cos(1.0472)),zed], 
    #                      [middle,middle+(4*spacing),zed],[middle,middle+(3*spacing),zed],[middle,middle+(2*spacing),zed],[middle,middle+spacing,zed], 
    #                      [middle+((spacing)*np.sin(1.0472)),middle-((spacing)*np.cos(1.0472)),zed],[middle+((spacing*2)*np.sin(1.0472)),middle-((spacing*2)*np.cos(1.0472)),zed],[middle+((spacing*3)*np.sin(1.0472)),middle-((spacing*3)*np.cos(1.0472)),zed],[middle+((spacing*4)*np.sin(1.0472)),middle-((spacing*4)*np.cos(1.0472)),zed]])
    
    return positions

def update_plot(frame):
    Locations = np.zeros((12,3))
    for x in range(12):
        Locations[x] = array_place(x+1)
    ax.scatter(Locations[:,0], Locations[:,1], Locations[:,2], marker = 'o', color = 'blue')
    x1, y1, z1 = TDOA(data_value,writepointer,[2,5,8,10])
    print(x1, y1, z1)
    ax.scatter(x1,y1,z1, marker = 's', color = 'red')
    

signal.signal(signal.SIGINT, signal_handler)


#PhaseDiff_Thread2 = threading.Timer(5,phase_difference)
#PhaseDiff_Thread2.daemon = True
#PhaseDiff_Thread2.start()

# Thread creation and start
RECV_NODE = threading.Thread(target=nodeA)
RECV_NODE.daemon = True
RECV_NODE.start()


ani = FuncAnimation(fig, update_plot, interval=100 ,cache_frame_data=False)
plt.show()




               
