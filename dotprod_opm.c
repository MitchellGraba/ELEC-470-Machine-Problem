// Mitchell Graba 20056482 OpenMP dotproduct calculator
#include <omp.h>
#include <stdio.h>

int main()
{
  int myid, t, i, n;

  printf("How many threads?: ");
  if (scanf("%d", &t) < 1)
  {
    return -1;
  }

  printf("Vector length = ");
  if (scanf("%d", &n) < 1)
  {
    printf("Check input for vector length.\n");
    return -1;
  }

  double a[n], b[n], result;

  /* Some initializations */
  result = 0.0;
  for (i = 0; i < n; i++)
  {
    a[i] = i;
    b[i] = i;
  }

  omp_set_num_threads(t);

#pragma omp parallel for default(shared) schedule(dynamic) reduction(+ \
                                                                     : result)

  for (i = 0; i < n; i++)
  {
    result += (a[i] * b[i]);
    printf("Thread %d, working at index %d \n\n", omp_get_thread_num(), i);
  }

  printf("Dot product= %f\n", result);
}