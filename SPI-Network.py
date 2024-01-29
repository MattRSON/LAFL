# Imports Libraries
import spidev
import RPi.GPIO as GPIO
import time

# Use the pin numbers on the board
GPIO.setmode(GPIO.BOARD)

# Chip select pin definitions
Select1 = 25

GPIO.setup(Select1, GPIO.OUT)

# Use Spi bus 0
bus = 0

# Chip select pin (Will probably be ignored for other pins)
device = 1

# Enable spi
spi = spidev.SpiDev()

# Open a connections to the device
spi.open(bus, device)

# Set the SPI speed and mode
spi.max_speed_hz = 1000000 #1Mhz
spi.mode = 0


while True:
    # try to read 16 bits from the spi bus
    GPIO.output(Select1, GPIO.HIGH) # Test fake chip select pin
    Data1 = spi.readbytes(2)
    ADC = Data1[0]*256+Data1[1]
    GPIO.output(Select1, GPIO.LOW)
    print(ADC)
    time.sleep(1)