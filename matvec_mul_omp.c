// Mitchell Graba 20056482 OpenMP matrix vector multiplication calculator
// This implements both methods of parallelization
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXTHRDS 64
#define PRINT 0
#define VALIDATE 1

// validates result with single threaded procedural computaion of multiplication
int validate(int m, int n, double **mat, double *vec, double *res)
{
  double temp = 0.0;
  for (int j = 0; j < m; j++)
  {
    for (int i = 0; i < n; i++)
    {
      temp += (mat[j][i] * vec[i]);
    }
    if (temp != res[j])
    {
      fprintf(stderr, "results incorrect\n");
      return -1;
    }
    temp = 0.0;
  }
  printf("results correct\n");
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

int setParams(int *meth, int *t, int *m, int *n, int argc, char *argv[])
{
  if (argc == 5)
  {
    *meth = strtol(argv[1], (char **)NULL, 10);
    *t = strtol(argv[2], (char **)NULL, 10);
    *m = strtol(argv[3], (char **)NULL, 10);
    *n = strtol(argv[3], (char **)NULL, 10);

    printf("TESTING:\tMethod: %d\t%d Threads\tm=%d\tn=%d\n", *meth, *t, *m, *n);
    if (*t > MAXTHRDS || *m > __INT_MAX__ || *n > __INT_MAX__ || !(*meth == 1 || *meth == 2))
    {
      fprintf(stderr, "error with parameters\n");
      return -1;
    }
  }
  else
  {
    printf("Which parallelization method?: ");
    if (scanf("%d", meth) < 1)
    {
      fprintf(stderr, "Check input for parallelization method.\n");
      return -1;
    }
    printf("How many threads?: ");
    if (scanf("%d", t) < 1)
    {
      fprintf(stderr, "Check input for number of processors.\n");
      return -1;
    }

    printf("How many rows?: ");
    if (scanf("%d", m) < 1)
    {
      fprintf(stderr, "Check input for row length.\n");
      return -1;
    }

    printf("How many columns?: ");
    if (scanf("%d", n) < 1)
    {
      fprintf(stderr, "Check input for column length.\n");
      return -1;
    }
  }
}

void matvec_mul1(int chunksize, int m, int n, double **mat, double *vec, double *res)
{
  int i, j;
  double t1, t2, elapsed = 0.0;

  //  Just the inner loop should be parallelized
  for (j = 0; j < m; j++)
  {
#if PRINT
    printf("Outerloop: Thread %d\n\n", omp_get_thread_num());
#endif
    t1 = omp_get_wtime();
#pragma omp parallel for default(shared) schedule(dynamic, chunksize) reduction(+ \
                                                                                : res[:m])
    for (i = 0; i < n; i++)
    {
      res[j] += (mat[j][i] * vec[i]);
#if PRINT
      printf("Innerloop: Thread %d, working on row %d column %d \n\n", omp_get_thread_num(), j, i);
#endif
    }
    
    t2 = omp_get_wtime();
    elapsed += t2 - t1;
  }
  printf("Program Executed in %fms\n", (elapsed)*1000.0);
}

void matvec_mul2(int m, int n, double **mat, double *vec, double *res)
{
  int i, j;
  double t1, t2;
  t1 = omp_get_wtime();
#pragma omp parallel for default(shared) schedule(dynamic) private(i) // each thread gets it's own private i
  for (j = 0; j < m; j++)
  {
#if PRINT
    printf("Outerloop: Thread %d\n\n", omp_get_thread_num());
#endif
    for (i = 0; i < n; i++)
    {
      res[j] += (mat[j][i] * vec[i]); // no reduction necessary
#if PRINT
      printf("Innerloop: Thread %d, working on row %d column %d \n\n", omp_get_thread_num(), j, i);
#endif
    }
  }
#pragma omp barrier
  t2 = omp_get_wtime();

  printf("Program Executed in %fms\n", (t2 - t1) * 1000.0);
}

int main(int argc, char *argv[])
{
  int myid, t, i, j, m, n, meth;

  setParams(&meth, &t, &m, &n, argc, argv);

  double **mat = (double **)malloc((m) * sizeof(double));
  double *vec = (double *)malloc((n) * sizeof(double));
  double *result = (double *)malloc((m) * sizeof(double));

  init(m, n, mat, vec, result);
  omp_set_num_threads(t);

  meth == 1 ? matvec_mul1(m / t, m, n, mat, vec, result) : meth == 2 ? matvec_mul2(m, n, mat, vec, result)
                                                                     : exit(0);

#if PRINT
  printf("[");
  for (i = 0; i < m; i++)
    printf("%f,", result[i]);
  printf("\b]T\n");
#endif
#if VALIDATE
  validate(m, n, mat, vec, result);
#endif
}
