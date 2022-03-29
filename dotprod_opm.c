#include <omp.h>
#include <stdio.h>

int main() {
  int myid;
  long i, n, chunk;
  printf("Vector length = ");
  if (scanf("%ld", &n) < 1) {
    printf("Check input for vector length.\n");
    return -1;
  }

  double a[n], b[n], result;

  /* Some initializations */
  chunk = 10;
  result = 0.0;
  for (i = 0; i < n; i++) {
    a[i] = 0.5 * i;
    b[i] = 2.0 * i;
  }

#pragma omp parallel for default(shared) private(i) schedule(static,chunk) reduction(+:result)

  for (i = 0; i < n; i++) {
    result += (a[i] * b[i]);
    if (omp_get_thread_num() == 0 && i == 0) {
      printf("Number of threads = %d\n", omp_get_num_threads());
    }
  }
  

  printf("Dot product= %f\n", result);
}