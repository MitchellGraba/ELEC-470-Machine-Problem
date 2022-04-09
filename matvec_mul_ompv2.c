// Mitchell Graba 20056482 OpenMP matrix vector multiplication calculator
// All threads compute their own row no reduction required
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define PRINT 0
#define VALIDATE 0

// validates result with single threaded procedural computaion of multiplication
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

int main()
{
  int myid, t, i, j, m, n;
  double t1, t2, elapsed = 0.0;

  printf("How many threads?: ");
  if (scanf("%d", &t) < 1)
  {
    return -1;
  }

  printf("How many rows?: ");
  if (scanf("%d", &m) < 1)
  {
    printf("Check input for row length.\n");
    return -1;
  }

  printf("How many columns?: ");
  if (scanf("%d", &n) < 1)
  {
    printf("Check input for column length.\n");
    return -1;
  }

  double **mat = (double **)malloc((m) * sizeof(double));
  for (i = 0; i < n; i++)
    mat[i] = (double *)malloc(n * sizeof(double));

  double *vec = (double*)malloc((n) * sizeof(double));

  for (i = 0; i < n; i++)
    vec[i] = i % 100;

  double *result = (double*)malloc((m) * sizeof(double));

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
  t1 = omp_get_wtime();

#pragma omp parallel for default(shared) schedule(dynamic) private(i) // each thread gets it's own private i
  for (j = 0; j < m; j++)
  {
#if PRINT
    printf("Outerloop: Thread %d\n\n", omp_get_thread_num());
#endif
    for (i = 0; i < n; i++)
    {
      result[j] += (mat[j][i] * vec[i]); // no reduction necessary
#if PRINT
      printf("Innerloop: Thread %d, working on row %d column %d \n\n", omp_get_thread_num(), j, i);
#endif
    }
  }
#pragma omp barrier
  t2 = omp_get_wtime();
#if PRINT
  printf("[");
  for (i = 0; i < m; i++)
    printf("%f,", result[i]);
  printf("\b]T\n");
#endif
#if VALIDATE
  validate(m, n, mat, vec, result);
#endif
  printf("Program Executed in %fms\n", (t2 - t1) * 1000.0);
}
