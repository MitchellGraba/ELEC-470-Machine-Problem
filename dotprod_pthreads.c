#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>

#define MAXTHRDS 64

typedef struct
{
  long double *x_th;            // memory address of where to start from in vec x
  long double *y_th;            // memory address of where to start from in vec y
  long double partial_dot_prod; // variable local to thread holding partial dotprod
  long double *global_dot_prod; // address of result array global to all threads
  pthread_mutex_t *mutex;
  int vec_len_th; // vector length for a given thread
} dot_product_thrd;

void *serial_dot_product(void *arg)
{
  // pointer to struct that contains the data a thread needs to do work
  dot_product_thrd *dot_data = arg;

  for (int i = 0; i < dot_data->vec_len_th; i++)
  {
    dot_data->partial_dot_prod += dot_data->x_th[i] * dot_data->y_th[i];
    // printf("Thread %d, working at index %d\n",syscall(__NR_gettid),i);
  }
  clock_t t1 = clock();
  pthread_mutex_lock(dot_data->mutex); // beginning of critical section
  *(dot_data->global_dot_prod) += dot_data->partial_dot_prod;
  pthread_mutex_unlock(dot_data->mutex); // end of critical section
  clock_t t2 = clock();
  printf("Thread %d critical section executed in %fms\n", syscall(__NR_gettid), 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC);
  pthread_exit(NULL); // thread is finished
}

int main()
{

  long double *x, *y, dot_prod;
  pthread_t *working_thread;
  dot_product_thrd *thrd_dot_prod_data;
  void *status;
  pthread_mutex_t *mutex_dot_prod;
  int num_of_thrds;
  int vec_len;
  int subvec_len;
  int i;

  printf("Number of threads = ");
  if (scanf("%d", &num_of_thrds) < 1 || num_of_thrds > MAXTHRDS)
  {
    printf("Check input for number of processors.\n");
    return -1;
  }
  printf("Vector length = ");
  if (scanf("%d", &vec_len) < 1)
  {
    printf("Check input for vector length.\n");
    return -1;
  }
  subvec_len = vec_len / num_of_thrds;

  dot_prod = 0.0;

  x = malloc(vec_len * sizeof(long double));
  y = malloc(vec_len * sizeof(long double));
  for (i = 0; i < vec_len; i++)
  {
    x[i] = i;
    y[i] = i;
  }

  working_thread = malloc(num_of_thrds * sizeof(pthread_t));
  thrd_dot_prod_data = malloc(num_of_thrds * sizeof(dot_product_thrd));
  mutex_dot_prod = malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(mutex_dot_prod, NULL);

  clock_t t1 = clock();
  for (i = 0; i < num_of_thrds; i++)
  {
    thrd_dot_prod_data[i].x_th = x + i * subvec_len;
    thrd_dot_prod_data[i].y_th = y + i * subvec_len;
    thrd_dot_prod_data[i].global_dot_prod = &dot_prod;
    thrd_dot_prod_data[i].mutex = mutex_dot_prod;
    thrd_dot_prod_data[i].vec_len_th =
        (i == num_of_thrds - 1) ? vec_len - (num_of_thrds - 1) * subvec_len
                                : subvec_len;
    pthread_create(&working_thread[i], NULL, serial_dot_product,
                   (void *)&thrd_dot_prod_data[i]);
    // printf("Thread %d dispatched working from %p to %p\n", i, x + i * subvec_len, (x + i * subvec_len) + thrd_dot_prod_data[i].vec_len_th - 1);
  }
  for (i = 0; i < num_of_thrds; i++)
    pthread_join(working_thread[i], &status);
  clock_t t2 = clock();
  printf("Dot product = %Lf\n", dot_prod);
  printf("Executed in %fms\n", 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC);

  free(x);
  free(y);
  free(working_thread);
  free(thrd_dot_prod_data);
  pthread_mutex_destroy(mutex_dot_prod);
  free(mutex_dot_prod);
}
