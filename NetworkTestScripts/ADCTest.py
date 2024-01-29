# Imports Libraries
import spidev
import RPi.GPIO as GPIO
import time

# Use the pin numbers on the board
GPIO.setmode(GPIO.BOARD)

# Chip select pin definitions
Select1 = 22

GPIO.setup(22, GPIO.OUT)

# Use Spi bus 0
bus = 0

# Chip select pin (Will be ignored for other pins)
device = 1

# Enable spi
spi = spidev.SpiDev()

# Open a connections to the device
spi.open(bus, device)
GPIO.output(22, GPIO.HIGH) # Enable fake CS pin

# Set the SPI speed and mode
spi.max_speed_hz = 1000000 #1Mhz
spi.mode = 0

while True:
    # try to read 16 bits from the spi bus
    GPIO.output(22, GPIO.LOW)
    Data1 = spi.readbytes(2)
    ADC = Data1[0]*256+Data1[1]
    GPIO.output(22, GPIO.HIGH)
    conn.sendall(ADC.to_bytes(2,'little'))
    print(ADC.to_bytes(2,'little'))
    time.sleep(1/50000)
