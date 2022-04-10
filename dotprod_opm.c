// Mitchell Graba 20056482 OpenMP dotproduct calculator
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXTHRDS 64

int validate(int len, double *x, double *y, double res)
{
  double temp;

  for (int i = 0; i < len; i++)
  {
    temp += (x[i] * y[i]);
  }
  if (temp != res)
  {
    fprintf(stderr, "results incorrect\n");
    return -1;
  }
  printf("results correct\n");
  return 1;
}

int main()
{
  int myid, t, i, vec_len, chunksize;
  double t1, t2;

  printf("How many threads?: ");
  if (scanf("%d", &t) < 1 || t > MAXTHRDS)
  {
    return -1;
  }

  printf("Vector length = ");
  if (scanf("%d", &vec_len) < 1)
  {
    printf("Check input for vector length.\n");
    return -1;
  }

  double *x = (double *)malloc((sizeof(double) * vec_len));
  double *y = (double *)malloc((sizeof(double) * vec_len));
  double dot_prod = 0.0;

  /* Some initializations */
  for (i = 0; i < vec_len; i++)
  {
    x[i] = i % 100;
    y[i] = i % 100;
  }

  omp_set_num_threads(t);
  t1 = omp_get_wtime();
#pragma omp parallel for default(shared) schedule(dynamic, ((vec_len / t) > 1) ? vec_len / t : 1) reduction(+ \
                                                                                                            : dot_prod)

  for (i = 0; i < vec_len; i++)
  {
    dot_prod += (x[i] * y[i]);
    printf("Thread %d, working at index %d \n\n", omp_get_thread_num(), i);
  }
#pragma omp barrier

  t2 = omp_get_wtime();
  printf("Dot product= %f\n", dot_prod);
  printf("Program Executed in %fms\n", (t2 - t1) * 1000.0);
  validate(vec_len, x, y, dot_prod);
}