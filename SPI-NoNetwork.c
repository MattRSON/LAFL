#include <stdio.h>
#include <stdlib.h>
#include <pigpio.h>
#include <time.h>
#include <unistd.h>

#define Select1 3
#define Select2 5
#define Select3 7
#define Select4 11
#define Select5 13
#define Select6 15
#define Select7 8
#define Select8 10
#define Select9 12
#define Select10 16
#define Select11 18
#define Select12 22


int main(){
    // Init gpio
    if (gpioInitialise()<0) return -1;
    int handle = 0;
    uint16_t Data[1];
    uint16_t BulkData[12];
    handle = spiOpen(1, 1000000, 0);
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
    struct timespec begin, end; 
    //clock_gettime(CLOCK_REALTIME, &begin);
    
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

    while(1){
        gpioWrite(Select1, 0);
        sleep(1e-6);
        spiRead(handle, (char*)Data, 2);
        gpioWrite(Select1, 1);
        printf("%s", (char*)Data);
        BulkData[0] = *Data;
        printf("%d ", BulkData[0]);
    }
    /*
    gpioWrite(Select2, 0);
    sleep(1e-6);
    spiRead(handle, (char*)Data,2);
    gpioWrite(Select2, 1);
    BulkData[1] = *Data;
    gpioWrite(Select3, 0);
    sleep(1e-6);
    spiRead(handle, (char*)Data,2);
    gpioWrite(Select3, 1);
    BulkData[2] = *Data;
    gpioWrite(Select4, 0);
    sleep(1e-6);
    spiRead(handle, (char*)Data,2);
    gpioWrite(Select4, 1);
    BulkData[3] = *Data;
    gpioWrite(Select5, 0);
    sleep(1e-6);
    spiRead(handle, (char*)Data,2);
    gpioWrite(Select5, 1);
    BulkData[4] = *Data;
    gpioWrite(Select6, 0);
    sleep(1e-6);
    spiRead(handle, (char*)Data,2);
    gpioWrite(Select6, 1);
    BulkData[5] = *Data;
    gpioWrite(Select7, 0);
    sleep(1e-6);
    spiRead(handle, (char*)Data,2);
    gpioWrite(Select7, 1);
    BulkData[6] = *Data;
    gpioWrite(Select8, 0);
    sleep(1e-6);
    spiRead(handle, (char*)Data,2);
    gpioWrite(Select8, 1);
    BulkData[7] = *Data;
    gpioWrite(Select9, 0);
    sleep(1e-6);
    spiRead(handle, (char*)Data,2);
    gpioWrite(Select9, 1);
    BulkData[8] = *Data;
    gpioWrite(Select10, 0);
    sleep(1e-6);
    spiRead(handle, (char*)Data,2);
    gpioWrite(Select10, 1);
    BulkData[9] = *Data;
    gpioWrite(Select11, 0);
    sleep(1e-6);
    spiRead(handle, (char*)Data,2);
    gpioWrite(Select11, 1);
    BulkData[10] = *Data;
    gpioWrite(Select12, 0);
    sleep(1e-6);
    spiRead(handle, (char*)Data,2);
    gpioWrite(Select12, 1);
    BulkData[11] = *Data;
    */
    
    // for (int i = 0; i < 12; i++) {
    //     printf("%d ", BulkData[i]);
    // }
    //sleep(1);
    //clock_gettime(CLOCK_REALTIME, &end);
    // double seconds = end.tv_sec - begin.tv_sec;
    // double nanoseconds = end.tv_nsec - begin.tv_nsec;
    // double elapsed = seconds + nanoseconds*1e-9;

    //printf("Time measured: %f seconds.\n", elapsed);

    // Start Loop
    /*
    while(1){
        gpioWrite(Select1, 0);
    }
    */

    // Terminate the library
    spiClose(handle);
    gpioTerminate();
    return 0;
}