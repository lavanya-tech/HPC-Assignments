#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000000

int main() {
    int i;
    float *a, sum = 0, avg, min = 1e9, max = -1e9;

    // Allocate memory for the array
    a = (float *) malloc(N * sizeof(float));

    // Initialize the array with some random values
    for (i = 0; i < N; i++) {
        a[i] = (float) rand() / RAND_MAX;
    }

    // Compute the sum, min, max, and average in parallel
#pragma omp parallel for reduction(+:sum) reduction(min:min) reduction(max:max)
    for (i = 0; i < N; i++) {
        sum += a[i];
        if (a[i] < min) {
            min = a[i];
        }
        if (a[i] > max) {
            max = a[i];
        }
    }
    avg = sum / N;

    printf("Sum: %f\n", sum);
    printf("Min: %f\n", min);
    printf("Max: %f\n", max);
    printf("Average: %f\n", avg);

    // Free memory
    free(a);

    return 0;
}
//gcc -fopenmp ArithmeticParallel.c -o reduction
//./reduction
