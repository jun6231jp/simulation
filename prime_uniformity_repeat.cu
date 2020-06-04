#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <time.h>

#define AVE_CHARGE 16 /* 各スレッドの平均分担量 */
#define THREAD_SIZE_X 1024
#define THREAD_SIZE_Y 1
#define THREAD_SIZE_Z 1
#define BLOCK_SIZE_X 1024
#define BLOCK_SIZE_Y 2
#define BLOCK_SIZE_Z 1
#define THREAD_NUM THREAD_SIZE_X * THREAD_SIZE_Y * THREAD_SIZE_Z * BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z
#define LIST AVE_CHARGE * THREAD_NUM 

__global__ void prime_cal( unsigned long long int *prime_result , unsigned long long int init_num ){

  unsigned long long int scan_idx = 0;
  unsigned long long int dev = 0;
  unsigned int flag = 0;

  /* 配列初期化 */
  for ( unsigned long long int list = 0 ; list < LIST ; list++ ){
    prime_result[list] = 0;
  }

  /*スレッドIDの割り当て*/
  unsigned long long int id = 
    ( blockIdx.z * BLOCK_SIZE_X * BLOCK_SIZE_Y + blockIdx.y * BLOCK_SIZE_X + blockIdx.x ) 
    * ( THREAD_SIZE_X * THREAD_SIZE_Y * THREAD_SIZE_Z ) + threadIdx.z * THREAD_SIZE_X * THREAD_SIZE_Y 
    + threadIdx.y * THREAD_SIZE_X + threadIdx.x;
  
  /* 計算量が全スレッドで均等になるよう配分 */
  unsigned long long int charge = ( unsigned long long int ) (2 / 3 * ( powf( double ( LIST ) , 1.5 ) - powf( double ( init_num ) , 1.5 ) )/ ( THREAD_NUM ) );
  unsigned long long int start_num[THREAD_NUM] = { 0 };
  unsigned long long int end_num[THREAD_NUM] = { 0 };
  start_num[0] = init_num;
  end_num[0] =( unsigned long long int ) powf( double ( 3 / 2 * charge + powf( double ( start_num[0] ) , 1.5 ) ) , 2 / 3 );
  for ( int i = 1 ; i < THREAD_NUM ; i++ ){
    start_num[i] = end_num[i-1] + 1;
    end_num[i] = ( unsigned long long int ) powf( double ( 3 / 2 * charge + powf( double ( start_num[i] ) , 1.5 ) ) , 2 / 3 );
  }
  
  /*素数判定を行う*/
  for ( scan_idx = start_num[id] ; scan_idx < end_num[id] + 1 ; scan_idx++ ) {
    flag=0;
    if ( scan_idx == 1 ){
      prime_result[scan_idx] = 0;
    }else if ( scan_idx == 2 ){
      prime_result[scan_idx] = 2;
    }else if ( scan_idx % 2 == 0 ){ 
      prime_result[scan_idx - init_num] = 0;
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
        prime_result[scan_idx - init_num] = scan_idx;
      }else if (flag == 1){
        prime_result[scan_idx - init_num] = 0;
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
  outputfile = fopen("./prime_data_cuda_uni_repeat.txt", "w");
  if (outputfile == NULL) {
    printf("cannot open\n");
    exit(1);
  }
  timer();

  /*ホスト側の変数設定*/
  unsigned long long int *host_result;
  unsigned long long int *host_result_2;

  /*デバイス側の変数設定*/
  unsigned long long int *prime_result;
  
  /*デバイスメモリ領域の確保*/
  checkCudaErrors(cudaMalloc((void**)&prime_result, LIST * sizeof(unsigned long long int) ));
  
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
  prime_cal<<<blocks , threads>>>(prime_result,0);
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
  host_result = (unsigned long long int*)malloc( LIST * sizeof(unsigned long long int) );
  checkCudaErrors(cudaMemcpy(host_result, prime_result, LIST * sizeof(unsigned long long int) , cudaMemcpyDeviceToHost));
  
  /*タイマーを停止し実行時間を表示*/
  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("Memory copy time: %f (msec)\n", msecTotal);

  /*デバイスメモリの開放*/
  checkCudaErrors(cudaFree(prime_result));

  /*デバイスメモリ領域の再確保*/
  checkCudaErrors(cudaMalloc((void**)&prime_result, LIST * sizeof(unsigned long long int) ));

  /*再度タイマー開始*/
  checkCudaErrors(cudaEventRecord(start, NULL));

  /*カーネルの起動*/
  prime_cal<<<blocks , threads>>>(prime_result,LIST);
  cudaThreadSynchronize();

  /*タイマーを停止し実行時間を表示*/
  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("Processing time: %f (msec)\n", msecTotal);
  
  /*再度タイマー開始*/
  checkCudaErrors(cudaEventRecord(start, NULL));
  
  /*結果の領域確保とデバイス側からのメモリ転送*/
  host_result_2 = (unsigned long long int*)malloc( LIST * sizeof(unsigned long long int) );
  checkCudaErrors(cudaMemcpy(host_result_2, prime_result, LIST * sizeof(unsigned long long int) , cudaMemcpyDeviceToHost));
  
  /*タイマーを停止し実行時間を表示*/
  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("Memory copy time: %f (msec)\n", msecTotal);

  printf("Now Writing...\n");
  for(unsigned long long int l = 0; l < LIST ; l++){ 
    if ( host_result[l] != 0 ){ 
      fprintf(outputfile,"%llu\n", host_result[l]);
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

