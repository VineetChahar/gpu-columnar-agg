#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(int* x){ x[threadIdx.x] = threadIdx.x*2; }
int main(){
  int n; cudaGetDeviceCount(&n);
  for(int d=0; d<n; ++d){ cudaDeviceProp p; cudaGetDeviceProperties(&p,d);
    printf("dev%d %s cc%d.%d SMs=%d smem/blk=%zu L2=%d MB gmem=%zu MB memclk=%d kHz busw=%d\n", d,p.name,p.major,p.minor,p.multiProcessorCount,p.sharedMemPerBlock,p.l2CacheSize>>20,p.totalGlobalMem>>20,p.memoryClockRate,p.memoryBusWidth);}
  int* d; cudaMalloc(&d,128); k<<<1,32>>>(d); int h[32]; cudaMemcpy(h,d,128,cudaMemcpyDeviceToHost);
  printf("kernel ok: h[31]=%d\n", h[31]); return h[31]==62 ? 0 : 1;
}
