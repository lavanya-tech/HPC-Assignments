#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 100000

void bubble_sort(float *a, int n) {
    int i, j;
    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j+1]) {
                float temp = a[j];
                a[j] = a[j+1];
                a[j+1] = temp;
            }
        }
    }
}

void parallel_bubble_sort(float *a, int n) {
    int i, j;
    #pragma omp parallel shared(a, n) private(i, j)
    {
        for (i = 0; i < n ; i++) {
            if(i%2 == 0){
            #pragma omp for
            for (j = 0; j < n - 1; j+=2) {
                if (a[j] > a[j+1]) {
                    float temp = a[j];
                    a[j] = a[j+1];
                    a[j+1] = temp;
                }
            }}
            else {
            #pragma omp for
            for (j = 0; j < n - 1; j+=2) {
                if (a[j] > a[j+1]) {
                    float temp = a[j];
                    a[j] = a[j+1];
                    a[j+1] = temp;
                }
            }
            }
        }
    }
}

int main() {
    int i, n = N;
    float *a, *b;
    double start_time, end_time;

    // Allocate memory for arrays
    a = (float *)malloc(n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));

    // Initialize array a with some data
    for (i = 0; i < n; i++) {
        a[i] = (float)rand() / RAND_MAX;
    }

    // Copy array a to array b
    for (i = 0; i < n; i++) {
        b[i] = a[i];
    }

    // Perform sequential bubble sort and measure performance
    start_time = omp_get_wtime();
    bubble_sort(a, n);
    end_time = omp_get_wtime();
    printf("Sequential bubble sort took %f seconds\n", end_time - start_time);

    // Perform parallel bubble sort and measure performance
    start_time = omp_get_wtime();
    parallel_bubble_sort(b, n);
    end_time = omp_get_wtime();
    printf("Parallel bubble sort took %f seconds\n", end_time - start_time);

    // Verify that the arrays are sorted correctly
    for (i = 0; i < n - 1; i++) {
        if (a[i] > a[i+1] || b[i] > b[i+1]) {
            printf("Error: arrays are not sorted correctly\n");
            break;
        }
    }

    // Free memory
    free(a);
    free(b);

    return 0;
}
//gcc -fopenmp ParallelBubblessort.c -o ParallelBubblessort
