import socket
import matplotlib.pyplot as plt
from collections import deque

# Constants
HOST = "LAFL"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

# Initialize plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
x_data = deque(maxlen=50)  # Keep track of the last 50 data points
y_data = deque(maxlen=50)
line, = ax.plot(x_data, y_data)

# Connect to the server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    
    while True:
        data = s.recv(2)
        if not data:
            break

        # Assuming the data received is a string representation of a number
        value = int(data.decode("utf-8"))

        # Update the data for plotting
        x_data.append(len(x_data) + 1)
        y_data.append(value)

        # Update the plot
        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.relim()
        ax.autoscale_view()

        # Redraw the plot
        plt.draw()
        plt.pause(0.1)  # Adjust the pause duration as needed