import socket
import time
import numpy as np
from timeit import default_timer as timer


HOST = ''
PORT = 65432

ADC = np.zeros(12)

frequency = 1 # Period in seconds
points = int(50000*(1/frequency))
Sinewave = np.zeros(points)
Cosinewave = np.zeros(points)
x_data = range(points)
i = 0

# Generate waveforms to send
for x in x_data:
    Sinewave[x] = int(((np.sin((x/50000)*2*(3.14)*(frequency)))+1)*2047)
    Cosinewave[x] = int(((np.cos((x/50000)*2*(3.14)*(frequency)))+1)*2047)

# Setup socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    # When connection is made start collecting and sending data
    with conn:
        while True:
            start = timer()
            
            ADC[0], ADC[1], ADC[2], ADC[3], ADC[4], ADC[5], ADC[6], ADC[7], ADC[8], ADC[9], ADC[10], ADC[11] = Sinewave[i], Cosinewave[i], 4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095

            i+=1

            if i>=(points):
                i = 0
                
            conn.sendall(bytes(ADC.astype(int)))
            end = timer()
            if (end-start) < (1/50000):
                time.sleep((1/50000)-(end-start))
            