// Mitchell Graba 20056482 OpenMP matrix vector multiplication calculator
// all threads work on the same row, then move on. reduction required.
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define PRINT 1
#define VALIDATE 1

// validates result with single threaded procedural computation of multiplication
int validate(int m, int n, double **mat, double *vec, double *res)
{
  double *t = (double *)malloc(sizeof(double));
  for (int j = 0; j < m; j++)
  {
    for (int i = 0; i < n; i++)
    {
      t[j] += (mat[j][i] * vec[i]);
    }
    if (t[j] != res[j])
    {
      printf("results incorrect\n");
      return -1;
    }
  }
  return 1;
}

void init(int m, int n, double **mat, double *vec, double *res)
{
  int i, j;

  for (i = 0; i < n; i++)
    mat[i] = (double *)malloc(n * sizeof(double));

  for (i = 0; i < n; i++)
    vec[i] = i % 100;

  /* Some initializations */

  for (i = 0; i < m; i++)
  {
    res[i] = 0.0;
  }

  for (i = 0; i < n; i++)
  {
    for (j = 0; j < m; j++)
    {
      mat[j][i] = (float)(1.0);
    }
  }
}

int main()
{
  int myid, t, i, j, m, n, chunksize;
  double red, t1, t2, elapsed = 0.0;

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
  double *vec = (double *)malloc((n) * sizeof(double));
  double *result = (double *)malloc((m) * sizeof(double));

  init(m, n, mat, vec, result);

  chunksize = m / t;
  red = 0.0;

  omp_set_num_threads(t);
#if PRINT
  printf("[");
#endif
  //  Just the inner loop should be parallelized
  for (j = 0; j < m; j++)
  {
    // printf("Outerloop: Thread %d\n\n", omp_get_thread_num());
    t1 = omp_get_wtime();
#pragma omp parallel for default(shared) schedule(dynamic, chunksize) reduction(+: red)
    for (i = 0; i < n; i++)
    {
      red += (mat[j][i] * vec[i]);
#if PRINT
      printf("Innerloop: Thread %d, working on row %d column %d \n\n", omp_get_thread_num(), j, i);
#endif
    }
#pragma omp barrier
    result[j] = red;
    red = 0.0;
    t2 = omp_get_wtime();
    elapsed += t2 - t1;
  }
#if PRINT
  for (i = 0; i < m; i++)
  {
    printf("%f,", result[i]);
  }
  printf("\b]T\n");
#endif
#if VALIDATE
  validate(m, n, mat, vec, result);
#endif
  printf("Program Executed in %fms\n", (elapsed)*1000.0);
}
