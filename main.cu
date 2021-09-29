
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <random>
#include <iostream>
#include <chrono>

# define CUDA_CHECK {\
    cudaError_t  cu_error = cudaGetLastError();                                 \
    if (cu_error != cudaSuccess) {                                              \
      std::cout << "Cuda error: " << cudaGetErrorString(cu_error) << std::endl; \
    }                                                                           \
  }

struct custom_functor{
    float factor;
    custom_functor(float _factor){
      factor = _factor;
    }
    __host__ __device__ int operator()(float &x) const {
        return (int) floor(x / factor);
    }
};


__global__ void custom_reduce_kernel(float *d_x, float *d_a, float *d_temp_a, int N, int Na, float da)
{
// Get our global thread ID
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float ix ;

  // Compute a
  for(int x = index; x < N; x += stride) {
      ix = floor(d_x[x] / da);

      d_temp_a[((int)ix) + Na * index] += 0.5;
      d_temp_a[((int)ix + 1) + Na * index] += 0.5;
  }
  __syncthreads();


  // Reduce
  for(int l = index; l < Na; l += stride) {
      for(int m = 0; m < stride; m += 1) {
          d_a[l] += d_temp_a[l + Na * m];
      }
  }
  __syncthreads();
}

void test_custom_reduce(float *d_x, float *d_a, float *h_a, int N, int Na, float da)
{
  int blockSize = 512; // Number of threads in each thread block
  int gridSize = (int)ceil((float) N /blockSize); // Number of thread blocks in grid

  // Create temp d_a array
  float *d_temp_a;
  cudaMalloc((void **) &d_temp_a, Na * blockSize * gridSize * sizeof(float));
  CUDA_CHECK;

  custom_reduce_kernel<<<gridSize,blockSize>>>(d_x, d_a, d_temp_a, N, Na, da);
  cudaMemcpy(h_a, d_a, Na * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_temp_a);
}



__global__ void thrust_reduce_kernel(float *d_a, int* d_a_idxs, int* d_a_cnts, int N, int Na, int n_entries)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= n_entries)
    return;

  int a_idx = d_a_idxs[index];
  int a_cnt = d_a_cnts[index];

  if ((a_idx + 1) >= Na || a_idx < 0 || a_idx >= Na || (a_idx + 1) < 0)
  {
    printf("Should not happen according to you!\n");
    return;
  }

  atomicAdd(&d_a[a_idx], a_cnt * 0.5f);
  atomicAdd(&d_a[a_idx+1], a_cnt * 0.5f);
}

void test_thrust_reduce(float *d_x, float *d_a, float *h_a, int N, int Na, float da)
{
  int *d_xi, *d_ones;
  int *d_a_cnt_keys, *d_a_cnt_vals;

  cudaMalloc((void**) &d_xi, N * sizeof(int));
  cudaMalloc((void**) &d_ones, N * sizeof(float));

  cudaMalloc((void**) &d_a_cnt_keys, Na * sizeof(int));
  cudaMalloc((void**) &d_a_cnt_vals, Na * sizeof(int));
  CUDA_CHECK;

  thrust::device_ptr<float> dt_x(d_x);
  thrust::device_ptr<float> dt_a(d_a);
  thrust::device_ptr<int> dt_xi(d_xi);
  thrust::device_ptr<int> dt_ones(d_ones);
  thrust::device_ptr<int> dt_a_cnt_keys(d_a_cnt_keys);
  thrust::device_ptr<int> dt_a_cnt_vals(d_a_cnt_vals);

  custom_functor f(da);
  thrust::fill(thrust::device, dt_a, dt_a + Na, 0.0f);
  thrust::fill(thrust::device, dt_ones, dt_ones + N, 1);
  thrust::fill(thrust::device, dt_a_cnt_keys, dt_a_cnt_keys + Na, -1);
  thrust::fill(thrust::device, dt_a_cnt_vals, dt_a_cnt_vals + Na, 0);

  thrust::transform(thrust::device, dt_x, dt_x + N, dt_xi, f);
  thrust::sort(thrust::device, dt_xi, dt_xi + N);

  thrust::pair<thrust::device_ptr<int>,thrust::device_ptr<int>> new_end;
  new_end = thrust::reduce_by_key(thrust::device, dt_xi, dt_xi + N, dt_ones, 
                                  dt_a_cnt_keys, dt_a_cnt_vals);

  int n_entries = new_end.first - dt_a_cnt_keys;
  int n_entries_2 = new_end.first - dt_a_cnt_keys;

  dim3 dimBlock(256);
  dim3 dimGrid((n_entries + dimBlock.x - 1) / dimBlock.x);
  thrust_reduce_kernel<<<dimGrid, dimBlock>>>(d_a, d_a_cnt_keys, d_a_cnt_vals, N, Na, n_entries);
  cudaMemcpy(h_a, d_a, Na * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_xi);
  cudaFree(d_ones);
  cudaFree(d_a_cnt_keys);
  cudaFree(d_a_cnt_vals);
}



__global__ void simple_atomicAdd_kernel(const float *d_x, float *d_a, float da, int N, int Na)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= N)
    return;

  int a_idx = floor(d_x[index] / da); // in principle i < size(a)

  atomicAdd(&d_a[a_idx], 0.5f);
  atomicAdd(&d_a[a_idx+1], 0.5f);
} 
void test_simple_atomicAdd(float *d_x, float *d_a, float *h_a, int N, int Na, float da)
{
  cudaMemset(d_a, 0, Na * sizeof(float));

  dim3 dimBlock(256);
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);
  simple_atomicAdd_kernel<<<dimGrid, dimBlock>>>(d_x, d_a, da, N, Na);
  cudaMemcpy(h_a, d_a, Na * sizeof(float), cudaMemcpyDeviceToHost);
}

void reference(float* h_x, float* h_a, int N, float da)
{
  for(int j = 0; j < N; j++) {
    float i = floor(h_x[j] / da); // in principle i < size(a)

    h_a[(int)i] += 0.5;
    h_a[(int)i+1] += 0.5; // I simplify the problem
  }
}

int main(int argc, char **argv)
{
  float da = 0.1f;
  int N = 100000;   
  int Na = 4096;   

  float L = 50; // box size
  float dxMesh = L / Na; // cell size

  float *h_x = (float *)malloc(N * sizeof(float));

  float *h_a1 = (float *)malloc(Na * sizeof(float));
  float *h_a2 = (float *)malloc(Na * sizeof(float));
  float *h_a3 = (float *)malloc(Na * sizeof(float));
  float *h_a_reference = (float *)malloc(Na * sizeof(float));

  /* Initialize random seed: */
  std::default_random_engine generator;
  std::uniform_real_distribution<float> generate_unif_dist(0.0,1.0);

  // h_x random initialisation
  for (int x = 0; x < N; x++) {
      float random = generate_unif_dist(generator);
      h_x[x] = random * L;
  }

  float *d_x, *d_a; 
  cudaMalloc((void**) &d_x, N * sizeof(float));
  cudaMalloc((void**) &d_a, Na * sizeof(float));

  cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();


  std::chrono::steady_clock::time_point current_bi = std::chrono::steady_clock::now();
  reference(h_x, h_a_reference, N, da);
  std::chrono::steady_clock::time_point current_ai = std::chrono::steady_clock::now();
  float time_ref = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();


  current_bi = std::chrono::steady_clock::now();
  test_custom_reduce(d_x, d_a, h_a1, N, Na, da);
  current_ai = std::chrono::steady_clock::now();
  float time1 = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();


  current_bi = std::chrono::steady_clock::now();
  test_thrust_reduce(d_x, d_a, h_a2, N, Na, da);
  current_ai = std::chrono::steady_clock::now();
  float time2 = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();

  current_bi = std::chrono::steady_clock::now();
  test_simple_atomicAdd(d_x, d_a, h_a3, N, Na, da);
  current_ai = std::chrono::steady_clock::now();
  float time3 = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();

  for (int i = 0; i < Na; i++)
  {
    if (fabs(h_a_reference[i]-h_a1[i]) > 0.0001)
      std::cout << "Error 1: " << i << " - " << h_a_reference[i] << " != " << h_a1[i] << std::endl;
    
    if (fabs(h_a_reference[i]-h_a2[i]) > 0.0001)
      std::cout << "Error 2: " << i << " - " << h_a_reference[i] << " != " << h_a2[i] << std::endl;
    
    if (fabs(h_a_reference[i]-h_a3[i]) > 0.0001)
      std::cout << "Error 3: " << i << " - " << h_a_reference[i] << " != " << h_a3[i] << std::endl;
  }

  std::cout << "Times: " << std::endl;
  std::cout << "- CPU Reference:         " << time_ref << " ms" << std::endl;
  std::cout << "- CUDA Custom reduce:    " << time1 << " ms" << std::endl;
  std::cout << "- CUDA Thrust reduce:    " << time2 << " ms" << std::endl;
  std::cout << "- CUDA Simple atomicAdd: " << time3 << " ms" << std::endl;

  free(h_x);
  free(h_a1);
  free(h_a2);
  free(h_a3);

  cudaFree(d_x);
  cudaFree(d_a);

  return 0;
}
