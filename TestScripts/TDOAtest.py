# Written by Mathew Rawson, Samuel Myers, Tyler LeMoine, Caise Taylor, and Robert Volkmann
# With assistance from Jai Bhullar of the Cyber Security and Computer Science department


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


DataRate = 10000 #Hz

fig, axs = plt.subplots(2, 2)
#fig, ax = plt.subplots() # Starts the plot

def phase_difference(input1, input2):
    

    F1, D1 = nodeFFT(input1, DataRate)
    F2, D2 = nodeFFT(input2, DataRate)
    
    # grab only the most prominent frequency 
    MaxD1 = np.max(abs(D1))
    IndmaxD1 = np.where(abs(D1) == MaxD1)[0][0]

    #print(F1[IndmaxD1])

    # Compute phase spectra
    phase_spectrum1 = np.angle(D1[IndmaxD1])
    print(phase_spectrum1)
    phase_spectrum2 = np.angle(D2[IndmaxD1])
    print(phase_spectrum2)

    # Calculate phase difference spectrum
    phase_diff_spectrum = phase_spectrum1 - phase_spectrum2
    
    
    #if phase_diff_spectrum < 0:
    #    phase_diff_spectrum = phase_diff_spectrum + (2 * np.pi)

    # Convert phase difference to time delay (optional)
    if F1[IndmaxD1] > 0:
        time_delay = phase_diff_spectrum / (2 * np.pi * F1[IndmaxD1])
    else:
        time_delay = 0

    return(phase_diff_spectrum, time_delay)

def nodeFFT(array,sampleRate):

    # Takes the real FFT of the data
    Fdomain = np.fft.rfft(array)
    Frequency = np.fft.rfftfreq(np.size(array),1/sampleRate)
    #print(F1[IndmaxD1])
    return(Frequency ,Fdomain)

# Drafting this Arrival time stuff
# For loop for finding sample when it hits threshold
def TDOA_cole(Data):
    # init index and flags
    samp_num = 1
    flag_1 = 0
    flag_2 = 0
    flag_3 = 0
    flag_4 = 0

    # init input#s
    input1 = Data[0]
    input2 = Data[1]
    input3 = Data[2]
    input4 = Data[3]
    D11 = nodeFFT(input1,7400)
    D21 = nodeFFT(input2,7400)
    D31 = nodeFFT(input3,7400)
    D41 = nodeFFT(input4,7400)
    print(D11,D21,D31,D41)
    while True:
        # check if flag not hit, check if signal at threshold, if so, mark flag and record sample number
        D1 = nodeFFT(np.concatenate((input1[samp_num+0:samp_num+25], np.zeros(50))),7400)
        D2 = nodeFFT(np.concatenate((input2[samp_num+0:samp_num+25], np.zeros(50))),7400)
        D3 = nodeFFT(np.concatenate((input3[samp_num+0:samp_num+25], np.zeros(50))),7400)
        D4 = nodeFFT(np.concatenate((input4[samp_num+0:samp_num+25], np.zeros(50))),7400)
        #print(D1/D11)

        if flag_1 == 0 and abs(D1) >= 3000:
                flag_1 = 1
                arrival_1 = samp_num
        elif flag_2 == 0 and abs(D2) >= 3000:
                flag_2 = 1
                arrival_2 = samp_num
        elif flag_3 == 0 and abs(D3) >= 3000:
                flag_3 = 1
                arrival_3 = samp_num
        elif flag_4 == 0 and abs(D4) >= 3000:
                flag_4 = 1
                arrival_4 = samp_num
        
        # iterative and break condition
        samp_num = samp_num+1
        if flag_1 == 1 and flag_2 == 1 and flag_3 ==1 and flag_4 == 1:
            # combine arrival times into array and convert to time values from sample number
            #arrival_times = [arrival_1*(1/DataRate), arrival_2*(1/DataRate), arrival_3*(1/DataRate), arrival_4*(1/DataRate)]
            arrival_times = [arrival_1, arrival_2, arrival_3, arrival_4]
            break

    return (arrival_times)


def equation(coords, Locations, relativeTime, SpeedOfSound):
    x1, y1, z1 = coords
    term1 = ((x1 - Locations[0][0])**2 + (y1 - Locations[0][1])**2 + (z1 - Locations[0][2])**2)**0.5
    term2 = ((x1 - Locations[1][0])**2 + (y1 - Locations[1][1])**2 + (z1 - Locations[1][2])**2)**0.5

    return term1 - term2 - SpeedOfSound * (relativeTime[0] - relativeTime[1])

def TDOA(Data, Inputs):
    input1 = butter_filter(Data[Inputs[0]-1])
    input2 = butter_filter(Data[Inputs[1]-1])
    input3 = butter_filter(Data[Inputs[2]-1])
    input4 = butter_filter(Data[Inputs[3]-1])


    SpeedOfSound = 345                                  # Speed of sound at high altitude in m/s

    timediffArray = TDOA_cole([input1,input2,input3,input4])
    print(timediffArray)

    Locations = np.zeros((4,3))
    for x in range(4):
        Locations[x] = array_place(Inputs[x])
    #print(Locations)
    initial_guess = [20, 20, 1]
    bounds = [(0, None), (0, None), (0, None)]
    result = minimize(equation, initial_guess, bounds=bounds, args=(Locations, timediffArray, SpeedOfSound))
    x1_opt, y1_opt, z1_opt = result.x
    print(x1_opt, y1_opt, z1_opt)
    return(1,2,3)

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
    # sos = [[ 2.13961520e-05,  4.27923041e-05,  2.13961520e-05,  1.00000000e+00, -1.44797260e+00,  7.75679511e-01],
    #        [ 1.00000000e+00,  2.00000000e+00,  1.00000000e+00, 1.00000000e+00, -1.37919243e+00,  7.98024930e-01],
    #        [ 1.00000000e+00,  0.00000000e+00, -1.00000000e+00,  1.00000000e+00, -1.56677322e+00,  8.33319209e-01],
    #        [ 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,  1.00000000e+00, -1.40423528e+00,  9.14112629e-01],
    #        [ 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,  1.00000000e+00, -1.69358308e+00,  9.37816513e-01]]
    
    sos = [[ 0.31761272,  0.63522543,  0.31761272,  1.,          1.08905692,  0.35395093],
           [ 1.,          2.,          1.,          1.,          1.37178046,  0.69698039],
           [ 1.,          0.,         -1.,          1.,         -0.4360535,  -0.47056428],
           [ 1.,         -2.,          1.,          1.,         -1.89890679,  0.90276295],
           [ 1.,         -2.,          1.,          1.,         -1.95866335,  0.96255059]]
    # Uses filter coefs to filter the data
    filtered = sosfilt(sos, data)
    return filtered

# Function for Microphone placements
def array_place(input):
    # Find midpoint of array
    zed = 1
    array_size = 50
    spacing = 0.1397
    middle = array_size/2
    if input == 1:
        positions = np.array([middle-((spacing*4)*np.sin(1.0472)),middle-((spacing*4)*np.cos(1.0472)),zed])
    if input == 2:
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

def IdealData():
    fakeSound = [20, 30, 1.5]
    SpeedOfSound = 345
    Locations = np.zeros((4,3))
    for x in range(4):
        Locations[x] = array_place(x+1)
    
    Distance1 = np.sqrt((fakeSound[0]-Locations[0,0])**2 + (fakeSound[1]-Locations[0,1])**2 + (fakeSound[2]-Locations[0,2])**2)
    Distance2 = np.sqrt((fakeSound[0]-Locations[1,0])**2 + (fakeSound[1]-Locations[1,1])**2 + (fakeSound[2]-Locations[1,2])**2)
    Distance3 = np.sqrt((fakeSound[0]-Locations[2,0])**2 + (fakeSound[1]-Locations[2,1])**2 + (fakeSound[2]-Locations[2,2])**2)
    Distance4 = np.sqrt((fakeSound[0]-Locations[3,0])**2 + (fakeSound[1]-Locations[3,1])**2 + (fakeSound[2]-Locations[3,2])**2)

    #print(Distance1, Distance2, Distance3, Distance4)

    T1 = Distance1/SpeedOfSound
    T2 = Distance2/SpeedOfSound
    T3 = Distance3/SpeedOfSound
    T4 = Distance4/SpeedOfSound

    P1 = T1*1000*2*np.pi
    P2 = T2*1000*2*np.pi
    P3 = T3*1000*2*np.pi
    P4 = T4*1000*2*np.pi

    points = int(1000) # Generates the list of points based on freqency and sample rate
    x_data = range(points)
    signal1 = np.zeros(points) # Initialize the list for a sine wave
    signal2 = np.zeros(points) # Initialize the list for a sine wave
    signal3 = np.zeros(points) # Initialize the list for a sine wave
    signal4 = np.zeros(points) # Initialize the list for a sine wave

    for x in x_data:
        
        signal1[x] = int(((np.sin(((x/DataRate)*2*(3.14)*(1000))+P1))+1)*1000)
        signal2[x] = int(((np.sin(((x/DataRate)*2*(3.14)*(1000))+P2))+1)*1000)
        signal3[x] = int(((np.sin(((x/DataRate)*2*(3.14)*(1000))+P3))+1)*1000)
        signal4[x] = int(((np.sin(((x/DataRate)*2*(3.14)*(1000))+P4))+1)*1000)
        

    return(x_data, signal1, signal2, signal3, signal4)
    
def update_plot(frame):
    global counter
    counter = counter + 1
    #print(counter)
    global input1
    F,D = nodeFFT(np.concatenate((input1[counter+0:counter+25], np.zeros(50))),10000)
    ax.plot(F[1:],abs(D[1:]))


def cross_correlation(signal1, signal2):
    # Reverse the second signal
    signal2_flipped = np.flip(signal2)
    # Calculate the cross-correlation
    return np.correlate(signal1, signal2_flipped, mode='full')

filelog = r'../Recorded Data 2/20240422_124549.csv'
print(filelog)
realData = np.loadtxt(filelog, delimiter=',')

# x, fakeData1, fakeData2, fakeData3, fakeData4 = IdealData()
# fakeData = [fakeData1,fakeData2,fakeData3,fakeData4]

# TDOA(realData, [2, 5, 8, 10])

#2,5,8,10

# input1 = np.concatenate((realData[:,1], np.zeros(100)))
# input2 = np.concatenate((realData[:,4], np.zeros(100)))
# input3 = np.concatenate((realData[:,7], np.zeros(100)))
# input4 = np.concatenate((realData[:,9], np.zeros(100)))

# input1 = butter_filter(realData[:,0])
# input2 = butter_filter(realData[:,3])
# input3 = butter_filter(realData[:,4])
# input4 = butter_filter(realData[:,6])






## Group A
input1 = realData[:,0]
input2 = realData[:,3]
input3 = realData[:,4]
input4 = realData[:,6]

## Group B
# input1 = realData[:,9]
# input2 = realData[:,4]
# input3 = realData[:,7]
# input4 = realData[:,1]

## Group C
# input1 = realData[:,2]
# input2 = realData[:,5]
# input3 = realData[:,8]
# input4 = realData[:,11]
#size = np.size(input1)-25

# counter = 0 
# for i in range(size):
#     MaxD = np.zeros(size)
#     IndmaxD = np.zeros(size)
#     F,D = nodeFFT(np.concatenate((input1[counter:counter+25], np.zeros(50))),10000)
#     MaxD[i] = np.max(abs(D))
#     IndmaxD[i] = np.where(abs(D) == MaxD[i])[0][0]

# MaxD2 = np.max(abs(MaxD))
# IndmaxD2 = np.where(abs(MaxD) == MaxD)[0][0]


#print(str(MaxD2) + " at location " + str(IndmaxD2))
# print(MaxD2)
# print(IndmaxD2)
# for x in input1:
#     print(x)
# input1 = butter_filter(fakeData1)
# input2 = butter_filter(fakeData2)
# input3 = ]butter_filter(fakeData3)
# input4 = butter_filter(fakeData4)
#print(np.size(input1))
# F1,D1 = nodeFFT(np.concatenate((input1, np.zeros(100))),DataRate)
# F2,D2 = nodeFFT(np.concatenate((input1, np.zeros(100))),DataRate)
# F3,D3 = nodeFFT(np.concatenate((input3, np.zeros(100))),DataRate)
# F4,D4 = nodeFFT(np.concatenate((input4, np.zeros(100))),DataRate)

#phase1,time1 = phase_difference(input1,input2)
#phase2,time2 = phase_difference(input2,input3)
#phase3,time3 = phase_difference(input3,input4)
#phase4,time4 = phase_difference(input4,input1)

#phase1,time1 = phase_difference(input1,input2)
#phase2,time2 = phase_difference(input2,input3)
#phase3,time3 = phase_difference(input3,input4)
#phase4,time4 = phase_difference(input4,input1)
#print(phase1, phase2, phase3, phase4)
#print(time1, time2, time3, time4)
#x1, y1, z1 = TDOA([input1, input2, input3, input4],[1,2,3,4])
#print(x1, y1, z1)

# axs[0, 0].plot(F1, abs(D1))
# axs[0, 0].set_title('F1')
# axs[0, 1].plot(F2, abs(D2), 'tab:orange')
# axs[0, 1].set_title('F2')
# axs[1, 0].plot(F3, abs(D3), 'tab:green')
# axs[1, 0].set_title('F3')
# axs[1, 1].plot(F4, abs(D4), 'tab:red')
# axs[1, 1].set_title('F4')
axs[0, 0].plot(input1)
axs[0, 0].set_title('Mic 1')
axs[0, 1].plot(input2, 'tab:orange')
axs[0, 1].set_title('Mic 4')
axs[1, 0].plot(input3, 'tab:green')
axs[1, 0].set_title('Mic 5')
axs[1, 1].plot(input4, 'tab:red')
axs[1, 1].set_title('Mic 7')


# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')

# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()  
  
#ani = FuncAnimation(fig, update_plot, interval=1 ,cache_frame_data=False)
plt.show()        



# import numpy as np

# def cross_correlation(signal1, signal2):
#     # Reverse the second signal
#     signal2_flipped = np.flip(signal2)
#     # Calculate the cross-correlation
#     return np.correlate(signal1, signal2_flipped, mode='full')

# # Example signals
# signal1 = np.array([1, 2, 3, 4, 5])
# signal2 = np.array([0, 1, 0.5])

# # Calculate cross-correlation
# result = cross_correlation(signal1, signal2)
# print("Cross-correlation result:", result)
