// Mitchell Graba 20056482 OpenMP matrix vector multiplication calculator
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  int myid;
  long i, j, m, n, chunk;

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
  chunk = 10;

  for (i = 0; i < m; i++) {
    result[i] = 0.0;
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      mat[j][i] = (float)(1.0); //(float)j / (i + 1);
    }
  }

  printf("[");

#pragma omp parallel for default(shared) private(i)
  for (j = 0; j < m; j++) {
    for (i = 0; i < n; i++) {
      result[j] += (mat[j][i] * vec[i]); // no reduction necessary
    }
    printf("%f,", result[j]);
  }

  printf("\b]T\n");
}
