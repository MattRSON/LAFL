import socket
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import signal
import sys
import numpy as np

HOST = "LAFL"   # Hostname
PORT = 65433    # Port

MAX_DATA_POINTS = 400

# Initialize plot
fig, ax = plt.subplots()
x_data = deque(maxlen=MAX_DATA_POINTS)
x_data = range(MAX_DATA_POINTS)
y_data = deque(maxlen=MAX_DATA_POINTS)
y_data.extend(np.ones(MAX_DATA_POINTS))
line, = ax.plot(x_data, y_data)

# Global Data Lock
data_lock = threading.Lock()
data_value = 0

# Thread to recieve data from PI (No Delay)
def nodeA():
    global data_value
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        while True:
            data = s.recv(2)
            if not data:
                break
            value = int.from_bytes(data, byteorder='big')
            with data_lock:
                data_value = value



# Function to update plot in animation
def update_plot(frame):
    with data_lock:
        value = data_value
    y_data.appendleft(value)
    y_data.pop
    line.set_xdata(x_data)
    line.set_ydata(y_data)

    ax.set_ylim(0,70000)

    ax.relim()
    ax.autoscale_view()

def signal_handler(sig, frame):
    print("ABORTING")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

ani = FuncAnimation(fig, update_plot, interval=1)

# Thread creation and start
RECV_NODE = threading.Thread(target=nodeA)
RECV_NODE.daemon = True
RECV_NODE.start()

plt.show()