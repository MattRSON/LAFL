# Written my Jai Bhullar and Mathew Rawson

import socket # For network connection
import threading # For threading used for plot
import matplotlib.pyplot as plt # For the plot
from matplotlib.animation import FuncAnimation # For continuous updates to the plot
from collections import deque # Data structure. Allows us to add and remove from both sides of the array
import signal # Used to safe shutdown the script
import sys # Also used to safe shutdown the script
import numpy as np # For extra number manipulation

## Setting up the network with the name of computer and what port its sending data on
HOST = "LAFL"   # Hostname
PORT = 65432    # Port

MAX_DATA_POINTS = 400 # 4 seconds of points

## Initialize plot
fig, ax = plt.subplots() # Starts the plot
x_data = deque(maxlen=MAX_DATA_POINTS) # Sets up the array givin the max points its allowed to have
x_data = range(MAX_DATA_POINTS) # Fills the array with numbers 0:Max points for the x axis
y_data = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
line, = ax.plot(x_data, y_data) # Graph the data points as a line

# Global Data Lock
data_lock = threading.Lock() # Prevents both threads from trying to modify a variable at the same time
data_value = 0 # Initializes the global data variable. This is the data from the ADCs

# Thread to recieve data from PI (No Delay)
def nodeA():
    global data_value # Grabs the global variable
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # Checks to see if the Rpi server is running
        s.connect((HOST, PORT)) # Tries to connect to the sever
        while True: # If it can
            data = s.recv(2) # Grab the data the server is sending
            if not data: # If the server stops sending data then close the connection
                break
            value = int.from_bytes(data, byteorder='big') # Turn the binary stream into an int
            with data_lock: # If this thread has control of the variable 
                data_value = value # Update it



# Function to update plot in animation
def update_plot(frame):
    with data_lock: # If the thread has control of the variable
        value = data_value # Grab the mose recent update
    y_data.appendleft(value) # Add it to the left of the array ei most recent data
    y_data.pop # Remove the right value in the array ei oldest data
    line.set_xdata(x_data) # Graph the x and y values as a line
    line.set_ydata(y_data)

    ax.set_ylim(0,4100) # Sets the limits of the graph

# Function to shutdown script safely
def signal_handler(sig, frame):
    print("ABORTING")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Sets how often the graph updates (The graph doesn't grab every value sent by the pi)
# Its to slow to do that
# But all of the data is still recived in real time and so we can process it quickly
ani = FuncAnimation(fig, update_plot, interval=1)

# Thread creation and start
RECV_NODE = threading.Thread(target=nodeA)
RECV_NODE.daemon = True
RECV_NODE.start()

# Show the plot
plt.show()