#include <stdio.h>
#include <pigpio.h>

#define Select1 = 3
#define Select2 = 5
#define Select3 = 7
#define Select4 = 11
#define Select5 = 13
#define Select6 = 15
#define Select7 = 8
#define Select8 = 10
#define Select9 = 12
#define Select10 = 16
#define Select11 = 18
#define Select12 = 22

int main(){
    // Init gpio
    if (gpioInitialise()<0) return -1;

    // Set pins as outputs
    gpioSetMode(Select1, PI_OUTPUT);
    gpioSetMode(Select2, PI_OUTPUT);
    gpioSetMode(Select3, PI_OUTPUT);
    gpioSetMode(Select4, PI_OUTPUT);
    gpioSetMode(Select5, PI_OUTPUT);
    gpioSetMode(Select6, PI_OUTPUT);
    gpioSetMode(Select7, PI_OUTPUT);
    gpioSetMode(Select8, PI_OUTPUT);
    gpioSetMode(Select9, PI_OUTPUT);
    gpioSetMode(Select10, PI_OUTPUT);
    gpioSetMode(Select11, PI_OUTPUT);
    gpioSetMode(Select12, PI_OUTPUT);

    // Writes all the pins high
    gpioWrite(Select1, 1);
    gpioWrite(Select2, 1);
    gpioWrite(Select3, 1);
    gpioWrite(Select4, 1);
    gpioWrite(Select5, 1);
    gpioWrite(Select6, 1);
    gpioWrite(Select7, 1);
    gpioWrite(Select8, 1);
    gpioWrite(Select9, 1);
    gpioWrite(Select10, 1);
    gpioWrite(Select11, 1);
    gpioWrite(Select12, 1);

    // Start Loop
    while(1){
        gpioWrite(Select1, 0);
    }

    // Terminate the library
    gpioTerminate();
    return 0;
}