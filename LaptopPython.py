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
#HOST = "127.0.0.1" # Loopback for HardwareEmulator.py
PORT = 65432    # Port

MAX_DATA_POINTS = 50 # 4 seconds of points

## Initialize plot
fig, ax = plt.subplots() # Starts the plot
x_data = deque(maxlen=MAX_DATA_POINTS) # Sets up the array givin the max points its allowed to have
x_data = range(MAX_DATA_POINTS) # Fills the array with numbers 0:Max points for the x axis
y_data1 = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data1.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
y_data2 = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data2.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
y_data3 = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data3.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
y_data4 = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data4.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
y_data5 = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data5.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
y_data6 = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data6.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
y_data7 = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data7.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
y_data8 = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data8.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
y_data9 = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data9.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
y_data10 = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data10.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
y_data11 = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data11.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
y_data12 = deque(maxlen=MAX_DATA_POINTS) # Sets up the array the same way as x_data
y_data12.extend(np.ones(MAX_DATA_POINTS)) # Fills the whole thing with ones to start
line1, = ax.plot(x_data, y_data1) # Graph the data points as a line
line2, = ax.plot(x_data, y_data2) # Graph the data points as a line
line3, = ax.plot(x_data, y_data3) # Graph the data points as a line
line4, = ax.plot(x_data, y_data4) # Graph the data points as a line
line5, = ax.plot(x_data, y_data5) # Graph the data points as a line
line6, = ax.plot(x_data, y_data6) # Graph the data points as a line
line7, = ax.plot(x_data, y_data7) # Graph the data points as a line
line8, = ax.plot(x_data, y_data8) # Graph the data points as a line
line9, = ax.plot(x_data, y_data9) # Graph the data points as a line
line10, = ax.plot(x_data, y_data10) # Graph the data points as a line
line11, = ax.plot(x_data, y_data11) # Graph the data points as a line
line12, = ax.plot(x_data, y_data12) # Graph the data points as a line

# Global Data Lock
data_lock = threading.Lock() # Prevents both threads from trying to modify a variable at the same time
data_value = np.zeros(12) # Initializes the global data variable. This is the data from the ADCs
value = np.zeros(12)

# Thread to receive data from PI (No Delay)
def nodeA():
    global data_value # Grabs the global variable
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # Checks to see if the Rpi server is running
        s.connect((HOST, PORT)) # Tries to connect to the sever
        while True: # If it can 
            packet = s.recv(48)
            
            bigint = int.from_bytes(packet,"big")
            value[0] = bigint & 0xffffffff
            value[1] = (bigint >> 32) & 0xffffffff
            value[2] = (bigint >> 64) & 0xffffffff
            value[3] = (bigint >> 96) & 0xffffffff
            value[4] = (bigint >> 128) & 0xffffffff
            value[5] = (bigint >> 160) & 0xffffffff
            value[6] = (bigint >> 192) & 0xffffffff
            value[7] = (bigint >> 224) & 0xffffffff
            value[8] = (bigint >> 256) & 0xffffffff
            value[9] = (bigint >> 288) & 0xffffffff
            value[10] = (bigint >> 320) & 0xffffffff
            value[11] = (bigint >> 352) & 0xffffffff
            print(value[0])
            with data_lock: # If this thread has control of the variable 
                data_value = value # Update it



# Function to update plot in animation
def update_plot(frame):
    with data_lock: # If the thread has control of the variable
        value = data_value # Grab the most recent update
    y_data1.appendleft(value[0]) # Add it to the left of the array ei most recent data
    y_data1.pop # Remove the right value in the array ei oldest data
    y_data2.appendleft(value[1]) # Add it to the left of the array ei most recent data
    y_data2.pop # Remove the right value in the array ei oldest data
    y_data3.appendleft(value[2]) # Add it to the left of the array ei most recent data
    y_data3.pop # Remove the right value in the array ei oldest data
    y_data4.appendleft(value[3]) # Add it to the left of the array ei most recent data
    y_data4.pop # Remove the right value in the array ei oldest data
    y_data5.appendleft(value[4]) # Add it to the left of the array ei most recent data
    y_data5.pop # Remove the right value in the array ei oldest data
    y_data6.appendleft(value[5]) # Add it to the left of the array ei most recent data
    y_data6.pop # Remove the right value in the array ei oldest data
    y_data7.appendleft(value[6]) # Add it to the left of the array ei most recent data
    y_data7.pop # Remove the right value in the array ei oldest data
    y_data8.appendleft(value[7]) # Add it to the left of the array ei most recent data
    y_data8.pop # Remove the right value in the array ei oldest data
    y_data9.appendleft(value[8]) # Add it to the left of the array ei most recent data
    y_data9.pop # Remove the right value in the array ei oldest data
    y_data10.appendleft(value[9]) # Add it to the left of the array ei most recent data
    y_data10.pop # Remove the right value in the array ei oldest data
    y_data11.appendleft(value[10]) # Add it to the left of the array ei most recent data
    y_data11.pop # Remove the right value in the array ei oldest data
    y_data12.appendleft(value[11]) # Add it to the left of the array ei most recent data
    y_data12.pop # Remove the right value in the array ei oldest data
    #plt.plot(x_data,y_data)
    #line.set_xdata(x_data) # Graph the x and y values as a line
    line1.set_ydata(y_data1)
    #line2.set_ydata(y_data2)
    #line3.set_ydata(y_data3)
    #line4.set_ydata(y_data4)
    #line5.set_ydata(y_data5)
    #line6.set_ydata(y_data6)
    #line7.set_ydata(y_data7)
    #line8.set_ydata(y_data8)
    #line9.set_ydata(y_data9)
    #line10.set_ydata(y_data10)
    #line11.set_ydata(y_data11)
    #line12.set_ydata(y_data12)

    ax.set_ylim(0,4100) # Sets the limits of the graph

# Function to shutdown script safely
def signal_handler(sig, frame):
    print("ABORTING")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Sets how often the graph updates (The graph doesn't grab every value sent by the pi)
# Its to slow to do that
# But all of the data is still received in real time and so we can process it quickly
cache_frame_data=False
ani = FuncAnimation(fig, update_plot, interval=100)

# Thread creation and start
RECV_NODE = threading.Thread(target=nodeA)
RECV_NODE.daemon = True
RECV_NODE.start()

# Show the plot
plt.show()