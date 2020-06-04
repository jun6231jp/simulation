#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <time.h>
#include <math.h>

#define Grid_x 256
#define Grid_y 256
#define Grid_z 2
#define Block_x 32
#define Block_y 32
#define Block_z 1

__global__ void thread_num( unsigned long int  *device_result );

/* timer */
int timer(void){
  time_t now = time(NULL);
  struct tm *pnow = localtime(&now);
  char buff[128]="";
  sprintf(buff,"%d:%d:%d",pnow->tm_hour,pnow->tm_min,pnow->tm_sec);
  printf("%s\n",buff);
  return 0;
}

int main(int argc, char** argv){
  FILE *outputfile;
  outputfile = fopen("./thread_num.txt", "w");
  if (outputfile == NULL) {
    printf("cannot open\n");
    exit(1);
  }
  timer();

  /*ブロックサイズとグリッドサイズの設定*/
  dim3 grid(Grid_x,Grid_y,Grid_z);
  dim3 block(Block_x,Block_y,Block_z);

  /*ホスト側の変数設定*/
  unsigned long int  thread_size = (Grid_x * Grid_y * Grid_z) * (Block_x * Block_y * Block_z);

  unsigned long int  *host_result;
  
  /*デバイス側の変数設定*/
  unsigned long int  *device_result;

  /*デバイスメモリ領域の確保*/
  checkCudaErrors(cudaMalloc((void**)&device_result, thread_size * sizeof(unsigned long int )));
  
  /*タイマーを作成して計測開始*/
  cudaEvent_t start;
  cudaEvent_t stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start, NULL));
  
  /*カーネルの起動*/
  thread_num<<<grid , block>>>(device_result);
  cudaThreadSynchronize();
  
  /*タイマーを停止し実行時間を表示*/
  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));
  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("Processing time: %f (msec)\n", msecTotal);
  
  /*再度タイマー開始*/
  checkCudaErrors(cudaEventRecord(start, NULL));
  
  /*結果の領域確保とデバイス側からのメモリ転送*/
  host_result = (unsigned long int *)malloc(thread_size * sizeof(unsigned long int ));
  checkCudaErrors(cudaMemcpy(host_result, device_result, thread_size * sizeof(unsigned long int ) , cudaMemcpyDeviceToHost));
  
  /*タイマーを停止し実行時間を表示*/
  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("Memory copy time: %f (msec)\n", msecTotal);
  printf("Now Writing...\n");
  printf("%u\n",thread_size);
  for( unsigned long int  l = 0; l < thread_size; l++ ){ 
    fprintf(outputfile,"%llu\n", host_result[l]);
  }
  fclose(outputfile);
  
  /*ホスト・デバイスメモリの開放*/
  free(host_result);
  checkCudaErrors(cudaFree(device_result));
  timer();
  
  /*終了処理*/
  cudaThreadExit();
  exit(0);
  
}

__global__ void thread_num(unsigned long int  *device_result){
  
  /*スレッドIDの割り当て*/
  /* blockDim = block size , threadIdx = 0~blockDim-1 , blockIdx = 0~grid size-1 , max thread = blockDim * max blockIdx + max threadIdx */
  unsigned long int  thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned long int  thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned long int  thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned long int thread_id = ( blockDim.x * (Grid_x - 1) + blockDim.x ) * ( blockDim.y * (Grid_y - 1) + blockDim.y ) * thread_idz + ( blockDim.x * (Grid_x - 1) + blockDim.x ) * thread_idy + thread_idx ;
  /*thread id*/
    device_result[thread_id] = thread_id;
}
