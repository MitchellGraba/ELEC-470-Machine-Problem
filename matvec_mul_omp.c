// Mitchell Graba 20056482 OpenMP matrix vector multiplication calculator
// all threads work on the same row, then move on. reduction required.
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
  int myid, t, i, j, m, n;

  printf("How many threads?: ");
  if (scanf("%d", &t) < 1)
  {
    return -1;
  }

  printf("How many rows?: ");
  if (scanf("%d", &m) < 1)
  {
    printf("Check input for vector length.\n");
    return -1;
  }

  printf("How many columns?: ");
  if (scanf("%d", &n) < 1)
  {
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

  for (i = 0; i < m; i++)
  {
    result[i] = 0.0;
  }

  for (i = 0; i < n; i++)
  {
    for (j = 0; j < m; j++)
    {
      mat[j][i] = (float)(1.0); 
    }
  }

  omp_set_num_threads(t);

  printf("[");
  // Just the inner loop should be parallelized
  for (j = 0; j < m; j++)
  {
    printf("Outerloop: Thread %d\n\n", omp_get_thread_num());

#pragma omp parallel for default(shared) schedule(dynamic) reduction(+ \
                                                                     : result[:m])
    for (i = 0; i < n; i++)
    {
      result[j] += (mat[j][i] * vec[i]);
      printf("Innerloop: Thread %d, working on row %d column %d \n\n", omp_get_thread_num(), j, i);
    }
#pragma omp barrier
  }
  for (i = 0; i < m; i++)
    printf("%f,", result[i]);
  printf("\b]T\n");
}
