#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <time.h>

#define Grid_x 1024
#define Grid_y 1024
#define Grid_z 1
#define Block_x 16
#define Block_y 8
#define Block_z 1


__global__ void thread_num(unsigned long long int  *device_result , unsigned long long int cycle);

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
  unsigned long long int num=0;
  unsigned long long int cycle = 0; 

  if (argv[1] == NULL) {
    printf("err\n");
    exit(1);
  }
  num=atoll(argv[1]);

  outputfile = fopen("./prime_num.txt", "w");
  if (outputfile == NULL) {
    printf("cannot open\n");
    exit(1);
  }
  timer();

  /*ブロックサイズとグリッドサイズの設定*/
  dim3 grid(Grid_x,Grid_y,Grid_z);
  dim3 block(Block_x,Block_y,Block_z);

  /*ホスト側の変数設定*/
  unsigned long long int thread_size = (Grid_x * Grid_y * Grid_z) * (Block_x * Block_y * Block_z);
  unsigned long long int *host_result;
  host_result = (unsigned long long int *)malloc(thread_size * sizeof(unsigned long long int));

  cycle = num / thread_size;

  /*デバイス側の変数設定*/
  unsigned long long int *device_result;

  for (unsigned long long int i = 0 ; i < cycle + 1 ; i++){
  
  /*デバイスメモリ領域の確保*/
  checkCudaErrors(cudaMalloc((void**)&device_result, thread_size * sizeof(unsigned long long int)));

  /*タイマーを作成して計測開始*/
  cudaEvent_t start;
  cudaEvent_t stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start, NULL));
  printf("Range : %llu - %llu\n",thread_size*i , thread_size*(i+1)-1);

  /*カーネルの起動*/
  thread_num<<<grid , block>>>(device_result, i);
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
  host_result = (unsigned long long int *)malloc(thread_size * sizeof(unsigned long long int));
  checkCudaErrors(cudaMemcpy(host_result, device_result, thread_size * sizeof(unsigned long long int ) , cudaMemcpyDeviceToHost));
  
  /*タイマーを停止し実行時間を表示*/
  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("Memory copy time: %f (msec)\n", msecTotal);
  printf("Now Writing...\n");

  for( unsigned long long int l = 0; l < thread_size; l++ ){ 
    if (host_result[l] != 0){
      fprintf(outputfile,"%llu\n", host_result[l]);
    }
  }
  /*ホスト・デバイスメモリの開放*/
  free(host_result);
  checkCudaErrors(cudaFree(device_result));
  }

  fclose(outputfile);
  timer();
  
  /*終了処理*/
  cudaThreadExit();
  exit(0);
}

__global__ void thread_num(unsigned long long int *device_result , unsigned long long int cycle){
  /*スレッドIDの割り当て*/
  /* メモ　: 
     blockDim = block size , 
     threadIdx = 0~blockDim-1 , 
     blockIdx = 0~grid size-1 , 
     max thread = blockDim * max blockIdx + max threadIdx 
  */
  unsigned long long int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned long long int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned long long int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned long long int thread_id = ( blockDim.x * (Grid_x - 1) + blockDim.x ) * ( blockDim.y * (Grid_y - 1) + blockDim.y ) * thread_idz + ( blockDim.x * (Grid_x - 1) + blockDim.x ) * thread_idy + thread_idx ;
  unsigned long long int dev = 0 ;
  unsigned long long int flag = 0;

  /*素数判定*/
  if ( thread_id == 1 ){
    device_result[thread_id] = 0;
  }else if ( thread_id == 2 ){
    device_result[thread_id] = 2;
  }else if ( thread_id % 2 == 0 ){ 
    device_result[thread_id] = 0;
  }else{
    dev = 3;
    while ( ( dev * dev ) <= thread_id + cycle * (Grid_x * Grid_y * Grid_z) * (Block_x * Block_y * Block_z)){
      if ( (thread_id + cycle * (Grid_x * Grid_y * Grid_z) * (Block_x * Block_y * Block_z) ) % dev == 0 ){ 
	flag=1;
	break;
      }
      dev += 2;
    }
    if (flag == 0){
      device_result[thread_id] = (unsigned long long int)(thread_id + cycle * (Grid_x * Grid_y * Grid_z) * (Block_x * Block_y * Block_z));
    }else if (flag == 1){
      device_result[thread_id] = 0;
    }
  }
}
