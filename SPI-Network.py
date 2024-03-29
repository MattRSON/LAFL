# Imports Libraries
import spidev
import RPi.GPIO as GPIO
import socket
import time

# Use the pin numbers on the board
GPIO.setmode(GPIO.BOARD)

# Chip select pin definitions
Select1 = 3

GPIO.setup(Select1, GPIO.OUT)

# Use Spi bus 0
bus = 0

# Chip select pin (Will be ignored for other pins)
device = 1

# Enable spi
spi = spidev.SpiDev()

# Open a connections to the device
spi.open(bus, device)
GPIO.output(Select1, GPIO.HIGH) # Enable fake CS pin

# Set the SPI speed and mode
spi.max_speed_hz = 1000000 #1Mhz
spi.mode = 0

# Setup tcp server
HOST = ''
PORT = 65433



# Setup socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    # When connection is made start collecting and sending data
    with conn:
        while True:
            # try to read 16 bits from the spi bus and send it over network
            GPIO.output(Select1, GPIO.LOW)
            Data1 = spi.readbytes(2)
            ADC = Data1[0]*256+Data1[1]
            GPIO.output(Select1, GPIO.HIGH)
            conn.sendall(ADC.to_bytes(2,'big'))
            #print(ADC.to_bytes(2,'little'))
            time.sleep(1/50000)
