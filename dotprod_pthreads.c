#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <sys/time.h>

#define MAXTHRDS 64
#define PRINT 0

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

long double current_time_ms(void)
{
  struct timespec spec;

  clock_gettime(CLOCK_REALTIME, &spec);

  return (1000.0 * spec.tv_sec + (spec.tv_nsec / 1.0e6));
}

typedef struct
{
  double *x_th;            // memory address of where to start from in vec x
  double *y_th;            // memory address of where to start from in vec y
  double partial_dot_prod; // variable local to thread holding partial dotprod
  double *global_dot_prod; // address of result array global to all threads
  pthread_mutex_t *mutex;
  int vec_len_th; // vector length for a given thread
} dot_product_thrd;

void *serial_dot_product(void *arg)
{
  long double t1, t2;
  // pointer to struct that contains the data a thread needs to do work
  dot_product_thrd *dot_data = arg;

  for (int i = 0; i < dot_data->vec_len_th; i++)
  {
    dot_data->partial_dot_prod += dot_data->x_th[i] * dot_data->y_th[i];
#if PRINT
    printf("Thread %d, working at index %d\n", syscall(__NR_gettid), i);
#endif
  }
  t1 = current_time_ms();
  pthread_mutex_lock(dot_data->mutex); // beginning of critical section
  *(dot_data->global_dot_prod) += dot_data->partial_dot_prod;
  pthread_mutex_unlock(dot_data->mutex); // end of critical section
  t2 = current_time_ms();
#if PRINT
  printf("Thread %d critical section executed in %Lfms\n", syscall(__NR_gettid), (t2 - t1));
#endif

  pthread_exit(NULL); // thread is finished
}

int main()
{

  double *x, *y, dot_prod;
  pthread_t *working_thread;
  dot_product_thrd *thrd_dot_prod_data;
  void *status;
  pthread_mutex_t *mutex_dot_prod;
  int num_of_thrds;
  int vec_len;
  int subvec_len;
  int i;
  long double t1, t2;

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

  x = malloc(vec_len * sizeof(double));
  y = malloc(vec_len * sizeof(double));
  for (i = 0; i < vec_len; i++)
  {
    x[i] = i % 100;
    y[i] = i % 100;
  }

  working_thread = malloc(num_of_thrds * sizeof(pthread_t));
  thrd_dot_prod_data = malloc(num_of_thrds * sizeof(dot_product_thrd));
  mutex_dot_prod = malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(mutex_dot_prod, NULL);

  t1 = current_time_ms();
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
#if PRINT
    printf("Thread %d dispatched working from %p to %p\n", i, x + i * subvec_len, (x + i * subvec_len) + thrd_dot_prod_data[i].vec_len_th - 1);
#endif
  }
  for (i = 0; i < num_of_thrds; i++)
    pthread_join(working_thread[i], &status);
  t2 = current_time_ms();
  printf("Dot product = %f\n", dot_prod);
  printf("Executed in %Lfms\n", (t2 - t1));
  validate(vec_len, x, y, dot_prod);
  free(x);
  free(y);
  free(working_thread);
  free(thrd_dot_prod_data);
  pthread_mutex_destroy(mutex_dot_prod);
  free(mutex_dot_prod);
}
