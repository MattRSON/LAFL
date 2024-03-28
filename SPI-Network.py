# Imports Libraries
import spidev
import RPi.GPIO as GPIO
import socket
import time
from timeit import default_timer as timer
import numpy as np
import sys
from datetime import datetime

# Use the pin numbers on the board
GPIO.setmode(GPIO.BOARD)

# Chip select pin definitions
Select1 = 3
Select2 = 5
Select3 = 7
Select4 = 11
Select5 = 13
Select6 = 15
Select7 = 8
Select8 = 10
Select9 = 12
Select10 = 16
Select11 = 18
Select12 = 22

GPIO.setwarnings(False)
GPIO.setup(Select1, GPIO.OUT)
GPIO.setup(Select2, GPIO.OUT)
GPIO.setup(Select3, GPIO.OUT)
GPIO.setup(Select4, GPIO.OUT)
GPIO.setup(Select5, GPIO.OUT)
GPIO.setup(Select6, GPIO.OUT)
GPIO.setup(Select7, GPIO.OUT)
GPIO.setup(Select8, GPIO.OUT)
GPIO.setup(Select9, GPIO.OUT)
GPIO.setup(Select10, GPIO.OUT)
GPIO.setup(Select11, GPIO.OUT)
GPIO.setup(Select12, GPIO.OUT)

# Use Spi bus 0
bus = 0

# Chip select pin (Will be ignored for other pins)
device = 1

# Enable spi
spi = spidev.SpiDev()

# Open a connections to the device
spi.open(bus, device)
GPIO.output(Select1, GPIO.HIGH) # Enable CS1 pin
GPIO.output(Select2, GPIO.HIGH) # Enable CS2 pin
GPIO.output(Select3, GPIO.HIGH) # Enable CS3 pin
GPIO.output(Select4, GPIO.HIGH) # Enable CS4 pin
GPIO.output(Select5, GPIO.HIGH) # Enable CS5 pin
GPIO.output(Select6, GPIO.HIGH) # Enable CS6 pin
GPIO.output(Select7, GPIO.HIGH) # Enable CS7 pin
GPIO.output(Select8, GPIO.HIGH) # Enable CS8 pin
GPIO.output(Select9, GPIO.HIGH) # Enable CS9 pin
GPIO.output(Select10, GPIO.HIGH) # Enable CS10 pin
GPIO.output(Select11, GPIO.HIGH) # Enable CS11 pin
GPIO.output(Select12, GPIO.HIGH) # Enable CS12 pin

# Set the SPI speed and mode
spi.max_speed_hz = 12000000 #12Mhz
spi.mode = 0

# Setup tcp server
HOST = ''
PORT = 65432


ADC = np.zeros(12) # Initialize the list to hold the 12 signals

DataRate = 50000

archive = np.zeros((12,DataRate))
tempArchive = np.zeros((12,DataRate))
counter = 0

# Setup socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    # When connection is made start collecting and sending data
    with conn:
        while True:
            start = timer()
            # try to read 16 bits from the spi bus and send it over network
            GPIO.output(Select1, GPIO.LOW)
            Data = spi.readbytes(2)
            ADC[0] = Data[0]*256+Data[1]
            GPIO.output(Select1, GPIO.HIGH)

            GPIO.output(Select2, GPIO.LOW)
            Data = spi.readbytes(2)
            ADC[1] = Data[0]*256+Data[1]
            GPIO.output(Select2, GPIO.HIGH)

            #GPIO.output(Select3, GPIO.LOW)
            #Data = spi.readbytes(2)
            #ADC[2] = Data[0]*256+Data[1]
            #GPIO.output(Select3, GPIO.HIGH)

            #GPIO.output(Select4, GPIO.LOW)
            #Data = spi.readbytes(2)
            #ADC[3] = Data[0]*256+Data[1]
            #GPIO.output(Select4, GPIO.HIGH)

            #GPIO.output(Select5, GPIO.LOW)
            #Data = spi.readbytes(2)
            #ADC[4] = Data[0]*256+Data[1]
            #GPIO.output(Select5, GPIO.HIGH)

            #GPIO.output(Select6, GPIO.LOW)
            #Data = spi.readbytes(2)
            #ADC[5] = Data[0]*256+Data[1]
            #GPIO.output(Select6, GPIO.HIGH)

            #GPIO.output(Select7, GPIO.LOW)
            #Data = spi.readbytes(2)
            #ADC[6] = Data[0]*256+Data[1]
            #GPIO.output(Select7, GPIO.HIGH)

            #GPIO.output(Select8, GPIO.LOW)
            #Data = spi.readbytes(2)
            #ADC[7] = Data[0]*256+Data[1]
            #GPIO.output(Select8, GPIO.HIGH)

            #GPIO.output(Select9, GPIO.LOW)
            #Data = spi.readbytes(2)
            #ADC[8] = Data[0]*256+Data[1]
            #GPIO.output(Select9, GPIO.HIGH)

            #GPIO.output(Select10, GPIO.LOW)
            #Data = spi.readbytes(2)
            #ADC[9] = Data[0]*256+Data[1]
            #GPIO.output(Select10, GPIO.HIGH)

            #GPIO.output(Select11, GPIO.LOW)
            #Data = spi.readbytes(2)
            #ADC[10] = Data[0]*256+Data[1]
            #GPIO.output(Select11, GPIO.HIGH)

            #GPIO.output(Select12, GPIO.LOW)
            #Data = spi.readbytes(2)
            #ADC[11] = Data[0]*256+Data[1]
            #GPIO.output(Select12, GPIO.HIGH)

            #ADC[1] = 0
            ADC[2] = 0
            ADC[3] = 0
            ADC[4] = 0
            ADC[5] = 0
            ADC[6] = 0
            ADC[7] = 0
            ADC[8] = 0
            ADC[9] = 0
            ADC[10] = 0
            ADC[11] = 0
            

            archive[:,counter] = ADC
            counter += 1
            archiveSize = int(np.size(archive)/12)
            if counter >= archiveSize:
                tempArchive = archive
                archive = np.zeros((12,(archiveSize+DataRate)))
                archive[:,0:archiveSize] = tempArchive
                tempArchive = np.zeros((12,(archiveSize+DataRate)))


            try:
                # Send all the data over the network as 32bit ints
                conn.sendall(bytes(ADC.astype(int)))
            except ConnectionResetError:
                currentDateTime = datetime.now()
                filePath = (currentDateTime.strftime('%y%m%d_%H%M%S')) + '.bin'

                print('Saving data to ' + filePath)
                print('File size is ' + str((archive.nbytes)/1000) + 'KB')
                archive.tofile(filePath)

                print("Shutdown")
                sys.exit(0)

            end = timer()
            
            # Delays based on how long it took to run the code.
            # This keeps the code running at the 50ksps rate
            #if (end-start) < (1/DataRate):
            #    time.sleep((1/DataRate)-(end-start))
            #else:
            #    print("oh no!") # If code is not keeping up we have a problem

        

