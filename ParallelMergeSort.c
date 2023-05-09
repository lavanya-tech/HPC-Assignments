#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 100000

void merge(float *a, int n, float *b, int m, float *c) {
    int i, j, k;
    i = 0; j = 0; k = 0;
    while (i < n && j < m) {
        if (a[i] < b[j]) {
            c[k] = a[i];
            i++;
        } else {
            c[k] = b[j];
            j++;
        }
        k++;
    }
    while (i < n) {
        c[k] = a[i];
        i++;
        k++;
    }
    while (j < m) {
        c[k] = b[j];
        j++;
        k++;
    }
}

void merge_sort(float *a, int n, float *b) {
    int i;
    if (n < 2) {
        return;
    }
    #pragma omp parallel sections shared(a, b, n)
    {
        #pragma omp section
        merge_sort(a, n/2, b);
        #pragma omp section
        merge_sort(a + n/2, n - n/2, b);
    }
    merge(a, n/2, a + n/2, n - n/2, b);
    for (i = 0; i < n; i++) {
        a[i] = b[i];
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

    // Perform sequential merge sort and measure performance
    start_time = omp_get_wtime();
    merge_sort(a, n, b);
    end_time = omp_get_wtime();
    printf("Sequential merge sort took %f seconds\n", end_time - start_time);

    // Perform parallel merge sort and measure performance
    start_time = omp_get_wtime();
    #pragma omp parallel shared(a, b, n)
    {
        #pragma omp single
        merge_sort(a, n, b);
    }
    end_time = omp_get_wtime();
    printf("Parallel merge sort took %f seconds\n", end_time - start_time);

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