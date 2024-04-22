// Written by Jai the c god (and Mathew a little bit)
#include <stdio.h>
#include <stdlib.h>
#include <pigpio.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#define Select1 2
#define Select2 3
#define Select3 4
#define Select4 17
#define Select5 27
#define Select6 22
#define Select7 14
#define Select8 15
#define Select9 18
#define Select10 23
#define Select11 24
#define Select12 25

FILE *fp;

int main(){
    // Init gpio
    if (gpioInitialise()<0) return -1;

    time_t rawtime;
    struct tm *timeinfo;
    char datetime_str[20]; // This will hold the formatted datetime string

    // Get current system time
    time(&rawtime);
    timeinfo = localtime(&rawtime);

    // Format the datetime string as per the specified format
    sprintf(datetime_str, "%04d%02d%02d_%02d%02d%02d",
            1900 + timeinfo->tm_year, // Year (1900 + current year)
            1 + timeinfo->tm_mon,     // Month (0-based index, so add 1)
            timeinfo->tm_mday,        // Day of the month
            timeinfo->tm_hour,        // Hour
            timeinfo->tm_min,         // Minute
            timeinfo->tm_sec);        // Second

    strcat(datetime_str,".csv");
    fp = fopen(datetime_str, "a");


    unsigned char Data[2];
    uint16_t BulkData[12];
    int handle = spiOpen(1, 6000000, 0);
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
        //printf("0x%02X", Data[0]); // This is for debugging 
        clock_gettime(CLOCK_REALTIME, &begin);
        gpioWrite(Select1, 0);
        //sleep(1e-6);
        spiRead(handle, Data, 2);
        gpioWrite(Select1, 1);
        BulkData[0] = (Data[0]*256)+Data[1];
        
        gpioWrite(Select2, 0);
        //sleep(1e-6);
        spiRead(handle, Data,2);
        gpioWrite(Select2, 1);
        BulkData[1] = (Data[0]*256)+Data[1];

        gpioWrite(Select3, 0);
        //sleep(1e-6);
        spiRead(handle, Data,2);
        gpioWrite(Select3, 1);
        BulkData[2] = (Data[0]*256)+Data[1];

        gpioWrite(Select4, 0);
        //sleep(1e-6);
        spiRead(handle, Data,2);
        gpioWrite(Select4, 1);
        BulkData[3] = (Data[0]*256)+Data[1];

        gpioWrite(Select5, 0);
        //sleep(1e-6);
        spiRead(handle, Data,2);
        gpioWrite(Select5, 1);
        BulkData[4] = (Data[0]*256)+Data[1];

        gpioWrite(Select6, 0);
        //sleep(1e-6);
        spiRead(handle, Data,2);
        gpioWrite(Select6, 1);
        BulkData[5] = (Data[0]*256)+Data[1];

        gpioWrite(Select7, 0);
        //sleep(1e-6);
        spiRead(handle, Data,2);
        gpioWrite(Select7, 1);
        BulkData[6] = (Data[0]*256)+Data[1];

        gpioWrite(Select8, 0);
        //sleep(1e-6);
        spiRead(handle, Data,2);
        gpioWrite(Select8, 1);
        BulkData[7] = (Data[0]*256)+Data[1];

        gpioWrite(Select9, 0);
        //sleep(1e-6);
        spiRead(handle, Data,2);
        gpioWrite(Select9, 1);
        BulkData[8] = (Data[0]*256)+Data[1];

        gpioWrite(Select10, 0);
        //sleep(1e-6);
        spiRead(handle, Data,2);
        gpioWrite(Select10, 1);
        BulkData[9] = (Data[0]*256)+Data[1];

        gpioWrite(Select11, 0);
        //sleep(1e-6);
        spiRead(handle, Data,2);
        gpioWrite(Select11, 1);
        BulkData[10] = (Data[0]*256)+Data[1];

        gpioWrite(Select12, 0);
        //sleep(1e-6);
        spiRead(handle, Data,2);
        gpioWrite(Select12, 1);
        BulkData[11] = (Data[0]*256)+Data[1];

        
        
        fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",BulkData[0],BulkData[1],BulkData[2],BulkData[3],BulkData[4],BulkData[5],BulkData[6],BulkData[7],BulkData[8],BulkData[9],BulkData[10],BulkData[11]);
        //fwrite(BulkData,16,12,fp);

        clock_gettime(CLOCK_REALTIME, &end);
        double seconds = end.tv_sec - begin.tv_sec;
        double nanoseconds = end.tv_nsec - begin.tv_nsec;
        double elapsed = seconds + nanoseconds*1e-9;

        if (elapsed < .0001){
            sleep(.0001-elapsed);
        } else {
            printf("Shits Fucked %f\n",elapsed);
        }
        //printf("Time measured: %f seconds.\n", elapsed);
        // for (int i = 0; i < 12; i++) {
        //     printf("%d ", BulkData[i]);
        // }
        // printf("\n");
        // sleep(1);
    }
   

    // Terminate the library
    fclose(fp);
    spiClose(handle);
    gpioTerminate();
    return 0;
}