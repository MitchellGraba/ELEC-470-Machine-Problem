// Mitchell Graba 20056482 OpenMP matrix vector multiplication calculator
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  int myid;
  long i, j, m, n;

  printf("How many rows?: ");
  if (scanf("%ld", &m) < 1) {
    printf("Check input for vector length.\n");
    return -1;
  }

  printf("How many columns?: ");
  if (scanf("%ld", &n) < 1) {
    printf("Check input for vector length.\n");
    return -1;
  }

  double **mat = (double **)malloc((m) * sizeof(double));
  for (i = 0; i < n; i++)
    mat[i] = (double *)malloc(n * sizeof(double));

  double vec[n];
  for (i = 0; i < n; i++)
    vec[i] = i;

  double result[m];

  /* Some initializations */

  for (i = 0; i < m; i++) {
    result[i] = 0.0;
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      mat[j][i] = (float)(1.0); //(float)j / (i + 1);
    }
  }

  printf("[");
// collapse 2 = parallelize two nested
#pragma omp parallel for default(shared) collapse(2) reduction(+ : result[:m])

  for (j = 0; j < m; j++) {
    for (i = 0; i < n; i++) {
      result[j] += (mat[j][i] * vec[i]);
    }
  }
  for (i = 0; i < m; i++)
    printf("%f,", result[i]);

  printf("\b]T\n");
}
