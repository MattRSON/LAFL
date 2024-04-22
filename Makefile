
all: build

build: SPI-NoNetwork.c
	git pull
	gcc SPI-NoNetwork.c -o temp -lpigpio
	chmod +x temp