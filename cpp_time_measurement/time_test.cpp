#include <stdio.h>
#include <time.h>
#include <sys/time.h> // needed for gettimeofday

void time_info()
{
    time_t t = time(NULL);
    printf("time_info()\n");
    printf("time() returns real time in seconds\n");
    printf("Current time by time() = %ld\n", t);
    printf("Size of time_t = %Zd\n", sizeof(t));
    printf("\n");
}

void clock_info()
{
    printf("clock_info()\n");
    printf("clock() returns CPU time in clocks\n");
    printf("CLOCKS_PER_SEC = %ld\n", CLOCKS_PER_SEC);
    printf("\n");
}

void timespec_info()
{
    printf("timespec_info()\n");
    printf("timespec_get() returns realtime in sec/nsec\n");
    printf("\n");
}

void gettimeofday_info()
{
    printf("gettimeofday_info()\n");
    printf("gettimeofday() returns realtime in sec/usec\n");
    printf("gettimeofday() is DEPRECATED!!!!\n");
    printf("\n");
}

void clock_gettime_info()
{
    printf("clock_gettime_info()\n");
    printf("clock_gettime()'s return value depend on clockid provided but in sec/nsec\n");
    printf("CLOCK_REALTIME: real time\n");
    printf("CLOCK_PROCESS_CPUTIME_ID: CPU time\n");
    printf("etc.\n");
    printf("\n");
}

void compare()
{
    clock_t c_start, c_end;
    time_t t_start, t_end;
    struct timespec ts_start, ts_end;
    struct timeval tv_start, tv_end;
    struct timespec cgr_start, cgr_end;
    struct timespec cgc_start, cgc_end;

    t_start = time(NULL);
    while(time(NULL) - t_start == 0);

    t_start = time(NULL);
    c_start = clock();
    timespec_get(&ts_start, TIME_UTC);
    gettimeofday(&tv_start, NULL);
    clock_gettime(CLOCK_REALTIME, &cgr_start);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cgc_start);

    while(time(NULL) - t_start < 2);

    t_end = time(NULL);
    c_end = clock();
    timespec_get(&ts_end, TIME_UTC);
    gettimeofday(&tv_end, NULL);
    clock_gettime(CLOCK_REALTIME, &cgr_end);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cgc_end);

    printf("time(): %ld\n", t_end - t_start);
    printf("clock(): %f\n", (double)(c_end - c_start) / CLOCKS_PER_SEC);
    printf("timespec_get(): %f\n",
           (double)(ts_end.tv_sec - ts_start.tv_sec) +
           (double)(ts_end.tv_nsec - ts_start.tv_nsec) * 1.0e-9);
    printf("gettimeofday(): %f\n",
           (double)(tv_end.tv_sec - tv_start.tv_sec) +
           (double)(tv_end.tv_usec - tv_start.tv_usec) * 1.0e-6);
    printf("clock_gettime(CLOCK_REALTIME): %f\n",
           (double)(cgr_end.tv_sec - cgr_start.tv_sec) +
           (double)(cgr_end.tv_nsec - cgr_start.tv_nsec) * 1.0e-9);
    printf("clock_gettime(CLOCK_PROCESS_CPUTIME_ID): %f\n",
           (double)(cgc_end.tv_sec - cgc_start.tv_sec) +
           (double)(cgc_end.tv_nsec - cgc_start.tv_nsec) * 1.0e-9);
}

/*
  Based on the following article
  https://www.mm2d.net/main/prog/c/time-01.html
 */
int main()
{
    time_info();
    clock_info();
    timespec_info();
    gettimeofday_info();
    clock_gettime_info();

    compare();
    
    return 0;
}
