import socket # For network connection
import time # Used for delaying the code to keep it on time
import numpy as np # For extra number manipulation
from timeit import default_timer as timer # Used to time the code to calculate the delay

## Setting up the network with the name of computer and what port its sending data on
HOST = '' # Hostname
PORT = 65432 # Port

# User Variables
frequency = 100 # frequency in Hz

# Fixed Variables
ADC = np.zeros(12) # Initialize the list to hold the 12 signals
points = int(50000*(1/frequency)) # Generates the list of points based on freqency and sample rate
Sinewave = np.zeros(points) # Initialize the list for a sine wave
Cosinewave = np.zeros(points) # Initialize the list for a cosine wave
x_data = range(points) # Generates the list of x points for all waveforms
i = 0 # Initialize the counter that keeps the system looping

DataRate = 50000

# Generate waveforms to send (More to come)
for x in x_data:
    Sinewave[x] = int(((np.sin((x/50000)*2*(3.14)*(frequency)))+1)*2047)
    Cosinewave[x] = int(((np.cos((x/50000)*2*(3.14)*(frequency)))+1)*2047)

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
            ADC[0] = Sinewave[i]
            ADC[1] = Cosinewave[i]
            ADC[2] = 4095
            ADC[3] = 4095
            ADC[4] = 4095
            ADC[5] = 4095
            ADC[6] = 4095
            ADC[7] = 4095
            ADC[8] = 4095
            ADC[9] = 4095
            ADC[10] = 4095
            ADC[11] = 4095

            # Increment the counter to keep the wave progressing
            i+=1

            # If the counter goes out of bounds then reset it
            if i>=(points):
                i = 0
            
            # Send all the data over the network as 32bit ints
            conn.sendall(bytes(ADC.astype(int)))
            end = timer() # Grabs the end time of the script

            # Delays based on how long it took to run the code.
            # This keeps the code running at the 50ksps rate
            if (end-start) < (1/50000):
                time.sleep((1/DataRate)-(end-start))
            