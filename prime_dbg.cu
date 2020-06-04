#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <time.h>

#define CHARGE 5
#define THREAD_SIZE_X 256
#define THREAD_SIZE_Y 64
#define THREAD_SIZE_Z 2
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 1
#define BLOCK_SIZE_Z 1
#define OFFSET 0
unsigned long long int list_num;

__global__ void prime_cal( unsigned long long int *prime_result , unsigned long long int offset){
/*スレッドIDの割り当て*/
int id = threadIdx.x
 + threadIdx.y * blockDim.x
 + blockIdx.x * blockDim.x * blockDim.y
 + blockIdx.y * blockDim.x * blockDim.y * gridDim.x
 + blockIdx.z * blockDim.x * blockDim.y * gridDim.x * gridDim.y;
  
 unsigned long long int scan_idx;
 unsigned long long int dev;
 unsigned long long int th_id;
 unsigned int flag;

/*素数判定を行う*/
  for ( scan_idx = id * CHARGE + offset; scan_idx < (id + 1) * CHARGE + offset; scan_idx++ ) {
    flag=0;
    th_id=scan_idx - offset;
    if ( scan_idx == 1 ){
      prime_result[th_id]=0;
    }else if ( scan_idx == 2 ){
      prime_result[th_id]=2;
    }else if ( scan_idx % 2 == 0 ){ 
      prime_result[th_id]=0;
    }else{
      dev=3;
      while ( (dev * dev) <= scan_idx ){
        if ( scan_idx % dev == 0 ){ 
          flag=1;
          break;
        }
        dev+=2;
      }
      if (flag == 0){
        prime_result[th_id]=scan_idx;
      }else if (flag == 1){
        prime_result[th_id]=0;
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
outputfile = fopen("./prime_data_cuda.txt", "w");
  if (outputfile == NULL) {
    printf("cannot open\n");
    exit(1);
  }
timer();
/*素数チェックする範囲を定義*/
list_num=CHARGE * THREAD_SIZE_X * THREAD_SIZE_Y * THREAD_SIZE_Z * BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;

/*ホスト側の変数設定*/
   unsigned long long int *host_result;

/*デバイス側の変数設定*/
   unsigned long long int *prime_result;

/*デバイスメモリ領域の確保*/
  checkCudaErrors(cudaMalloc((void**)&prime_result, list_num * sizeof(unsigned long long int) ));

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
  prime_cal<<<blocks , threads>>>(prime_result ,OFFSET);
  cudaThreadSynchronize();

/*タイマーを停止しかかった時間を表示*/
  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));
  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("Processing time: %f (msec)\n", msecTotal);

/*再度タイマー開始*/
  checkCudaErrors(cudaEventRecord(start, NULL));

/*結果の領域確保とデバイス側からのメモリ転送*/
  host_result = (unsigned long long int*)malloc( list_num * sizeof(unsigned long long int) );
  checkCudaErrors(cudaMemcpy(host_result, prime_result, list_num * sizeof(unsigned long long int) , cudaMemcpyDeviceToHost));

/*タイマーを停止しかかった時間を表示*/
  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("Memory copy time: %f (msec)\n", msecTotal);
  printf("Now Writing...\n");
  for(int l=0; l < list_num ; l++){ 
    if ( host_result[l] != 0 ){ 
      fprintf(outputfile,"%llu\n",host_result[l]);
    }
  }
  fclose(outputfile);
/*ホスト・デバイスメモリの開放*/
  free(host_result);
  checkCudaErrors(cudaFree(prime_result));
timer();
/*終了処理*/
  cudaThreadExit();
  exit(1);
}

