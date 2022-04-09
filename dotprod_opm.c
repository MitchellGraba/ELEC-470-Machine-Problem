// Mitchell Graba 20056482 OpenMP dotproduct calculator
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXTHRDS 64

int main()
{
  int myid, t, i, n;
  double t1, t2;

  printf("How many threads?: ");
  if (scanf("%d", &t) < 1 || t > MAXTHRDS)
  {
    return -1;
  }

  printf("Vector length = ");
  if (scanf("%d", &n) < 1)
  {
    printf("Check input for vector length.\n");
    return -1;
  }

  long double *a = (long double*)malloc((sizeof(long double) * n));
  long double *b = (long double*)malloc((sizeof(long double) * n));
  long double result = 0.0;

  /* Some initializations */
  for (i = 0; i < n; i++)
  {
    a[i] = i;
    b[i] = i;
  }

  omp_set_num_threads(t);
  t1 = omp_get_wtime();
#pragma omp parallel for default(shared) schedule(dynamic) reduction(+ \
                                                                     : result)

  for (i = 0; i < n; i++)
  {
    result += (a[i] * b[i]);
    //printf("Thread %d, working at index %d \n\n", omp_get_thread_num(), i);
  }
#pragma omp barrier

  t2 = omp_get_wtime();
  printf("Dot product= %Lf\n", result);
  printf("Program Executed in %fms\n", (t2-t1)*1000.0);
}