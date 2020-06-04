#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <time.h>

#define CHARGE 1 
#define THREAD_SIZE_X 256
#define THREAD_SIZE_Y 32
#define THREAD_SIZE_Z 1
#define BLOCK_SIZE_X 2
#define BLOCK_SIZE_Y 2
#define BLOCK_SIZE_Z 1

unsigned int list_num;

__global__ void prime_cal( unsigned int *prime_result){
/*スレッドIDの割り当て*/

unsigned int id = threadIdx.x
 + threadIdx.y * blockDim.x
 + blockIdx.x * blockDim.x * blockDim.y
 + blockIdx.y * blockDim.x * blockDim.y * gridDim.x
 + blockIdx.z * blockDim.x * blockDim.y * gridDim.x * gridDim.y;

  unsigned int scan_idx;
  unsigned int dev;
  unsigned int flag;
/*素数判定を行う*/
  for ( scan_idx = id * CHARGE ; scan_idx < (id + 1) * CHARGE; scan_idx++ ) {
    flag=0;
    if ( scan_idx == 1 ){
      prime_result[scan_idx]=0;
    }else if ( scan_idx == 2 ){
      prime_result[scan_idx]=2;
    }else if ( scan_idx % 2 == 0 ){ 
      prime_result[scan_idx]=0;
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
        prime_result[scan_idx]=scan_idx;
      }else if (flag == 1){
        prime_result[scan_idx]=0;
      }
    }
  }
    __syncthreads();
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
   unsigned int *host_result;

/*デバイス側の変数設定*/
   unsigned int *prime_result;

/*デバイスメモリ領域の確保*/
  checkCudaErrors(cudaMalloc((void**)&prime_result, list_num * sizeof(unsigned int) ));

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
  prime_cal<<<blocks , threads>>>(prime_result);
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
  host_result = (unsigned int*)malloc( list_num * sizeof(unsigned int) );
  checkCudaErrors(cudaMemcpy(host_result, prime_result, list_num * sizeof(unsigned int) , cudaMemcpyDeviceToHost));

/*タイマーを停止しかかった時間を表示*/
  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("Memory copy time: %f (msec)\n", msecTotal);
  printf("Now Writing...\n");
  for(int l=0; l < list_num ; l++){ 
    if ( host_result[l] != 0 ){ 
      fprintf(outputfile,"%u\n",host_result[l]);
    }
  }
  fclose(outputfile);
/*ホスト・デバイスメモリの開放*/
  free(host_result);
  checkCudaErrors(cudaFree(prime_result));
timer();
/*終了処理*/
  cudaThreadExit();
  return 0;
}

