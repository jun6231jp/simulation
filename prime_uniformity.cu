#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <time.h>
#include <math.h>
#define AVE_CHARGE 256 /* 各スレッドの平均分担量 */
#define THREAD_SIZE_X 1024
#define THREAD_SIZE_Y 16
#define THREAD_SIZE_Z 1
#define BLOCK_SIZE_X 2
#define BLOCK_SIZE_Y 2
#define BLOCK_SIZE_Z 1
#define thread_num THREAD_SIZE_X * THREAD_SIZE_Y * BLOCK_SIZE_X * BLOCK_SIZE_Y
#define list_num thread_num * AVE_CHARGE

__global__ void prime_cal( unsigned int *device_result , unsigned int *start_num , unsigned int *end_num );

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
  unsigned int str_size = list_num * sizeof(unsigned int);
  unsigned int thread_size = thread_num * sizeof(unsigned int);
  unsigned int *host_result;
  
  /*デバイス側の変数設定*/
  unsigned int *device_result;
  unsigned int *start_num;
  unsigned int *end_num;

  /* 計算量が全スレッドで均等になるよう配分 */
  unsigned int start_host[thread_num] = {0};
  unsigned int end_host[thread_num] = {0};
  double charge = AVE_CHARGE;
  start_host[0] = 2;
  //  end_host[0] = (unsigned int)charge + start_host[0];
  end_host[0] = 1 + start_host[0];
  for ( unsigned int i = 1 ; i < thread_num ; i++ ){
    if (start_host[i-1] == 0 ) {
      start_host[i]=0;
      end_host[i]=0;
    } else {
      start_host[i] = end_host[i-1]+1;
      //      end_host[i] =  (unsigned int)charge + start_host[i];
      end_host[i] = 1 + start_host[i];
      printf("%d , %d\n",start_host[i],end_host[i]);
    }
    if ( end_host[i] > list_num ){
      printf("%d > %d\n",end_host[i],list_num);
      start_host[i]=0;
      end_host[i]=0;
    }
  }
  /*デバイスメモリ領域の確保*/
  checkCudaErrors(cudaMalloc((void**)&device_result, str_size));
  checkCudaErrors(cudaMalloc((void**)&start_num, str_size));
  checkCudaErrors(cudaMalloc((void**)&end_num, str_size));
  checkCudaErrors(cudaMemcpy(start_num, start_host, thread_size , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(end_num, end_host, thread_size , cudaMemcpyHostToDevice));
  /*ブロックサイズとグリッドサイズの設定*/
  dim3 threadIdx(THREAD_SIZE_X,THREAD_SIZE_Y);
  dim3 blockDim(BLOCK_SIZE_X,BLOCK_SIZE_Y);
  
  /*タイマーを作成して計測開始*/
  cudaEvent_t start;
  cudaEvent_t stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start, NULL));
  
  /*カーネルの起動*/
  prime_cal<<<blockDim , threadIdx>>>(device_result,start_num,end_num);
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
  host_result = (unsigned int*)malloc(str_size);
  checkCudaErrors(cudaMemcpy(host_result, device_result, str_size , cudaMemcpyDeviceToHost));
  
  /*タイマーを停止し実行時間を表示*/
  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("Memory copy time: %f (msec)\n", msecTotal);
  printf("Now Writing...\n");
  for( unsigned int l = 0; l < list_num ; l++ ){ 
    if ( host_result[l] != 0 ){ 
      fprintf(outputfile,"%u\n", host_result[l]);
    }
  }
  fclose(outputfile);
  
  /*ホスト・デバイスメモリの開放*/
  free(host_result);
  checkCudaErrors(cudaFree(device_result));
  checkCudaErrors(cudaFree(start_num));
  checkCudaErrors(cudaFree(end_num));
  timer();
  
  /*終了処理*/
  cudaThreadExit();
  exit(0);
  
}

__global__ void prime_cal( unsigned int *device_result , unsigned int *start_num , unsigned int *end_num ){
  
  unsigned int dev = 0;
  unsigned int flag = 0;
  //  unsigned int list = 0;
  /* 配列初期化 */
  //for ( list = 0 ; list < list_num ; list++ ){
  //  device_result[list] = 0;
  // }
  /*スレッドIDの割り当て*/
  unsigned int thread_idx = threadIdx.x+BLOCK_SIZE_X*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+BLOCK_SIZE_Y*blockIdx.y;
  unsigned int thread_id = BLOCK_SIZE_X * THREAD_SIZE_X * thread_idy + thread_idx;
  
  /*素数判定を行う*/
  if (end_num[thread_id]==0) { 
   for ( unsigned int scan_idx = start_num[thread_id] ; scan_idx < end_num[thread_id] ; scan_idx++ ) {
     device_result[scan_idx] = 0;
   }
  } else {
    for ( unsigned int scan_idx = start_num[thread_id] ; scan_idx < end_num[thread_id] ; scan_idx++ ) {
      flag = 0;
      if ( scan_idx == 1 ){
	device_result[scan_idx] = 0;
      }else if ( scan_idx == 2 ){
	device_result[scan_idx] = 2;
      }else if ( scan_idx % 2 == 0 ){ 
	device_result[scan_idx] = 0;
      }else{
	dev = 3;
	while ( ( dev * dev ) <= scan_idx ){
	  if ( scan_idx % dev == 0 ){ 
	    flag=1;
	    break;
	  }
	  dev += 2;
	}
	if (flag == 0){
	  device_result[scan_idx] = scan_idx;
	}else if (flag == 1){
	  device_result[scan_idx] = 0;
	}
      }
      //	  device_result[scan_idx] = thread_id;
    }
  }
  //    __syncthreads();
}
