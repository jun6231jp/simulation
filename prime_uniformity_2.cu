#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <time.h>
#include <math.h>
#define AVE_CHARGE 5 /* 各スレッドの平均分担量 */
#define THREAD_SIZE_X 1024
#define THREAD_SIZE_Y 1
#define THREAD_SIZE_Z 1
#define BLOCK_SIZE_X 1024
#define BLOCK_SIZE_Y 1
#define BLOCK_SIZE_Z 1
#define thread_num THREAD_SIZE_X * THREAD_SIZE_Y * THREAD_SIZE_Z * BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z
#define list_num thread_num * AVE_CHARGE

__global__ void prime_cal( unsigned long long int *device_result ){

  unsigned long long int dev = 0;
  unsigned long long int flag = 0;
  unsigned long long int list = 0;
  /* 配列初期化 */
  for ( list = 0 ; list < list_num ; list++ ){
    device_result[list] = 0;
  }
  __syncthreads();
  /*スレッドIDの割り当て*/
  unsigned long long int thread_id = 
    ( blockIdx.z * BLOCK_SIZE_X * BLOCK_SIZE_Y + blockIdx.y * BLOCK_SIZE_X + blockIdx.x ) 
    * ( THREAD_SIZE_X * THREAD_SIZE_Y * THREAD_SIZE_Z ) + threadIdx.z * THREAD_SIZE_X * THREAD_SIZE_Y 
    + threadIdx.y * THREAD_SIZE_X + threadIdx.x;

  /* 計算量が全スレッドで均等になるよう配分 */
  unsigned long long int start_num = 0;
  unsigned long long int end_num = 0;
  double charge = powf(list_num,3/2)/thread_num;
  end_num = (unsigned long long int)powf(charge,2/3);
  for ( unsigned long long int i = 1 ; i < thread_id ; i++ ){
    start_num = (unsigned long long int)end_num+1;
    end_num = (unsigned long long int)powf(charge+powf(start_num,3/2),2/3);
  }
   __syncthreads();
  /*素数判定を行う*/
  for ( unsigned long long int scan_id = start_num ; scan_id < end_num + 1 ; scan_id++ ) {
    flag = 0;
    if ( scan_id == 1 ){
      device_result[scan_id] = 0;
    }else if ( scan_id == 2 ){
      device_result[scan_id] = 2;
    }else if ( scan_id % 2 == 0 ){ 
      device_result[scan_id] = 0;
    }else{
      dev = 3;
      while ( ( dev * dev ) <= scan_id ){
        if ( scan_id % dev == 0 ){ 
          flag=1;
          break;
        }
        dev += 2;
      }
      if (flag == 0){
        device_result[scan_id] = scan_id;
      }else if (flag == 1){
        device_result[scan_id] = 0;
      }
    }
    __syncthreads();
  }
}

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
  outputfile = fopen("./prime_data_cuda_uni.txt", "w");
  if (outputfile == NULL) {
    printf("cannot open\n");
    exit(1);
  }
  timer();
  printf("Max : %d\n",list_num);
  printf("Thread : %d\n",thread_num);
  /*ホスト側の変数設定*/
  unsigned long long int str_size = list_num * sizeof(unsigned long long int);
  unsigned long long int *host_result;

  /*デバイス側の変数設定*/
  unsigned long long int *device_result;
  
  /*デバイスメモリ領域の確保*/
  checkCudaErrors(cudaMalloc((void**)&device_result, str_size));

  /*ブロックサイズとグリッドサイズの設定*/
  dim3 threads(THREAD_SIZE_X,THREAD_SIZE_Y,THREAD_SIZE_Z);
  dim3 blocks(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  
  /*タイマーを作成して計測開始*/
  cudaEvent_t start;
  cudaEvent_t stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start, NULL));
  
  /*カーネルの起動*/
  prime_cal<<<blocks , threads>>>(device_result);
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
  host_result = (unsigned long long int*)malloc(str_size);
  checkCudaErrors(cudaMemcpy(host_result, device_result, str_size , cudaMemcpyDeviceToHost));
  
  /*タイマーを停止し実行時間を表示*/
  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("Memory copy time: %f (msec)\n", msecTotal);
  printf("Now Writing...\n");
  for( unsigned long long int l = 0; l < list_num ; l++ ){ 
    if ( host_result[l] != 0 ){ 
      fprintf(outputfile,"%llu\n", host_result[l]);
    }
  }
  fclose(outputfile);

  /*ホスト・デバイスメモリの開放*/
  free(host_result);
  checkCudaErrors(cudaFree(device_result));
  timer();

  /*終了処理*/
  cudaThreadExit();
  exit(1);

}

