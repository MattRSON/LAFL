import socket # For network connection
import time # Used for delaying the code to keep it on time
import numpy as np # For extra number manipulation
from timeit import default_timer as timer # Used to time the code to calculate the delay
import sys

## Setting up the network with the name of computer and what port its sending data on
HOST = '' # Hostname
PORT = 65432 # Port

# User Variables
frequency = 1 # frequency in Hz
DataRate = 50000

# Fixed Variables
ADC = np.zeros(12) # Initialize the list to hold the 12 signals
points = int(50000*(1/frequency)) # Generates the list of points based on freqency and sample rate
Sinewave = np.zeros(points) # Initialize the list for a sine wave
Cosinewave = np.zeros(points) # Initialize the list for a cosine wave
x_data = range(points) # Generates the list of x points for all waveforms
i = 0 # Initialize the counter that keeps the system looping




# Generate waveforms to send (More to come)
for x in x_data:
    Sinewave[x] = int(((np.sin((x/DataRate)*2*(3.14)*(frequency)))+1)*2047)
    Cosinewave[x] = int(((np.cos((x/DataRate)*2*(3.14)*(frequency)))+1)*2047)

realData = np.fromfile('240326_135933.bin', dtype=np.float64)
realData = np.reshape(realData,(12,-1))
points = np.size(realData)/12

# Setup socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # Checks to see if the laptop has connected
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept() # Accept the connection
    # When connection is made start collecting and sending data
    with conn:
        while True:
            start = timer() # Grabs the start time
            
            # Sets all the signals the the value that should be sent
            ADC[0] = realData[0,i] #Sinewave[i]
            ADC[1] = realData[1,i] #Cosinewave[i]
            ADC[2] = realData[2,i]
            ADC[3] = realData[3,i]
            ADC[4] = realData[4,i]
            ADC[5] = realData[5,i]
            ADC[6] = realData[6,i]
            ADC[7] = realData[7,i]
            ADC[8] = realData[8,i]
            ADC[9] = realData[9,i]
            ADC[10] = realData[10,i]
            ADC[11] = realData[11,i]

            # Increment the counter to keep the wave progressing
            i+=1

            # If the counter goes out of bounds then reset it
            if i>=(points):
                i = 0
            
            try:
                # Send all the data over the network as 32bit ints
                conn.sendall(bytes(ADC.astype(int)))
            except ConnectionResetError:

                print("Shutdown")
                sys.exit(0)

            end = timer() # Grabs the end time of the script

            # Delays based on how long it took to run the code.
            # This keeps the code running at the 50ksps rate
            if (end-start) < (1/DataRate):
                time.sleep((1/DataRate)-(end-start))  