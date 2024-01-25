import spidev

bus = 0

device = 1

spi = spidev.SpiDev()

spi.open(bus, device)

spi.max_speed_hz = 1000000
spi.mode = 0