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
  // Just the inner loop should be parrallized 
  for (j = 0; j < m; j++) {
    //printf("Outerloop: I am thread %d\n\n", omp_get_thread_num());

    #pragma omp parallel for default(shared) schedule(dynamic) reduction(+ : result[:m])
    for (i = 0; i < n; i++) {
      result[j] += (mat[j][i] * vec[i]);
      printf("Innerloop: I am thread %d, working on row %ld column %ld \n\n", omp_get_thread_num(), j, i);
    }
  }
  for (i = 0; i < m; i++)
    printf("%f,", result[i]);

  printf("\b]T\n");
}