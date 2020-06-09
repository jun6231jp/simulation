#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//TDB
//衝突角度や距離をスライドで変更できるようにする

#define PI 3.141592653589793
//物理パラメータ
#define cap 1000
#define ref 0.9//0.5
#define temp 4000
#define visc 9
#define GRAV (6.674*0.00000000000000000001)
#define density (2.5 * 1000000000000)
#define sigma (0.96*5.67*0.00000001) //W/m^2 T^4
#define cool (sigma*4*PI*rad*rad*1000000*10)
//粒子形状
#define rad 300//40 //km
#define M (4 / 3 * PI * rad*rad*rad* density)//kg
//描写設定
#define MOONOFFSET_X (INIT_WIDTH/vision*2)
#define MOONOFFSET_Y (INIT_WIDTH/vision*3)
#define MOONOFFSET_Z (INIT_HEIGHT/vision*3)
#define dev 360//12
#define resol 10
#define hollow 100//10
#define X 0
#define Y 1
#define Z 2
#define ANIM_START 0
#define ANIM 20//10
#define scale 0.01
#define colmargin 1.0001//1.0001
#define adjust 0.999
#define R (rad * scale)
#define INIT_WIDTH 800
#define INIT_HEIGHT 800
#define vision 40
#define Grid_x 2//block間は__syncthreadでは同期不可
#define Grid_y 2
#define Grid_z 1
#define Block_x 2//32
#define Block_y 2//16
#define Block_z 2//1

#define NUM_POINTS (Grid_x*Grid_y*Grid_z*Block_x*Block_y*Block_z)

unsigned int dev_points = dev + 1;
unsigned int window_width = INIT_WIDTH;
unsigned int window_height = INIT_HEIGHT;
double vision_size = vision;
float right_motion=0;
float up_motion=0;
double left, right, bottom, top;
float h_point[NUM_POINTS][3]={0};
float v_point[NUM_POINTS][3]={0};
float st_point[NUM_POINTS]={0};
float e_point[NUM_POINTS]={0};
float J_point[NUM_POINTS]={0};
float hv_buff[NUM_POINTS][3]={0};
float hp_buff[NUM_POINTS][3]={0};
float anim_time = ANIM_START;
float anim_dt = ANIM;
double phi = 30.0;
double theta = 30.0;
float light_pos[4]={0};
int mouse_old_x, mouse_old_y;
bool motion_p;
bool motion_w;
double eye[3]={0};
double center[3] = {0.0, 0.0, 0.0};
double up[3]={0};
double ** point;
float (*d_point)[3];
float (*dv_point)[3];
float (*dst_point);
float (*de_point);
float (*dJ_point);
float (*v_buff)[3];
float (*p_buff)[3];
float colsynctime[NUM_POINTS][NUM_POINTS]={0};
int colsyncindex[NUM_POINTS][NUM_POINTS]={0};
float (*dcolsynctime)[NUM_POINTS];
int (*dcolsyncindex)[NUM_POINTS];
__global__ void grav_coldetect(float(*pos)[3],float(*vec)[3],float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS],float(*p_buff)[3]);
__global__ void grav_padjust(float(*pos)[3], float(*p_buff)[3]);
__global__ void grav_colv(float(*pos)[3],float(*vec)[3],float(*v_buff)[3],float(*sti),float(*e),float(*J),float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS]);
__global__ void grav_v(float(*pos)[3],float(*vec)[3],float(*v_buff)[3],int(*colindex)[NUM_POINTS]);
__global__ void grav_vupdate(float(*vec)[3],float(*v_buff)[3]);
__global__ void buff_clear(float(*v_buff)[3],float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS],float(*p_buff)[3]);
__global__ void grav_p(float (*pos)[3], float(*vec)[3]);

//基本関数群
double dot(double vec0[], double vec1[])
{
  return(vec0[X] * vec1[X] + vec0[Y] * vec1[Y] + vec0[Z] * vec1[Z]);
}
void cross(double vec0[], double vec1[], double vec2[])
{
  vec2[X] = vec0[Y] * vec1[Z] - vec0[Z] * vec1[Y];
  vec2[Y] = vec0[Z] * vec1[X] - vec0[X] * vec1[Z];
  vec2[Z] = vec0[X] * vec1[Y] - vec0[Y] * vec1[X];
}
void normVec(double vec[])
{
  double norm;
  norm = sqrt(vec[X] * vec[X] + vec[Y] * vec[Y] + vec[Z] * vec[Z]);
  vec[X] /= norm;
  vec[Y] /= norm;
  vec[Z] /= norm;
}
void normal(double p0[], double p1[], double p2[], double normal[])
{
  unsigned int i;
  double v0[3], v1[3];
  for (i = 0; i < 3; i++) {
    v0[i] = p2[i] - p1[i];
    v1[i] = p0[i] - p1[i];
  }
  cross(v0, v1, normal);
  normVec(normal);
}

//衝突検知
__global__ void grav_coldetect(float(*pos)[3],float(*vec)[3],float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS],float(*p_buff)[3])
{
  float xn,yn,zn,vx,vy,vz,dis,sq;
  unsigned int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned int index = (blockDim.x * Grid_x) * (blockDim.y * Grid_y) * thread_idz + (blockDim.x * Grid_x) * thread_idy + thread_idx ;
  float rvec[3]={0};
  xn = pos[index][X];
  yn = pos[index][Y];
  zn = pos[index][Z];
  vx = vec[index][X];
  vy = vec[index][Y];
  vz = vec[index][Z];
  p_buff[index][X]=xn;
  p_buff[index][Y]=yn;
  p_buff[index][Z]=zn;
  for (int i = 0 ; i < NUM_POINTS; i++)
    {
      sq = (float)pow((double)(xn-pos[i][X]),2) + pow((double)(yn-pos[i][Y]),2) + pow((double)(zn-pos[i][Z]),2);
      dis = (float)sqrt((double)sq);
      rvec[X]=(pos[i][X]-xn)/dis;
      rvec[Y]=(pos[i][Y]-yn)/dis;
      rvec[Z]=(pos[i][Z]-zn)/dis;
      //衝突域侵入判定
      if (dis > 2 * R && i != index)
	{
	  colindex[index][i]=NUM_POINTS;
	}
      else if (dis <= 2 * R && i != index)
	{
	  //衝突域侵入からの経過の時間を記録　
	  colindex[index][i]=i;
	  coltime[index][i]=(2*R-dis)/((vx-vec[i][X])*rvec[X]+(vy-vec[i][Y])*rvec[Y]+(vz-vec[i][Z])*rvec[Z]);
	  //位置補正
	  if(dis <= 2 * R * adjust)
	    {
	      p_buff[index][X]+=(p_buff[index][X]-pos[i][X])/dis*(2*R*colmargin-dis);
	      p_buff[index][Y]+=(p_buff[index][Y]-pos[i][Y])/dis*(2*R*colmargin-dis);
	      p_buff[index][Z]+=(p_buff[index][Z]-pos[i][Z])/dis*(2*R*colmargin-dis);
	    }
	}
      else
	{
	  colindex[index][i]=NUM_POINTS;
	}
    }
}
//中心間距離が直径以下に近づいたものを補正
__global__ void grav_padjust(float(*pos)[3], float(*p_buff)[3])
{
  unsigned int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned int index = (blockDim.x * Grid_x) * (blockDim.y * Grid_y) * thread_idz + (blockDim.x * Grid_x) * thread_idy + thread_idx ;
  pos[index][X]=p_buff[index][X];
  pos[index][Y]=p_buff[index][Y];
  pos[index][Z]=p_buff[index][Z];
}
//衝突後の速度を計算
__global__ void grav_colv(float(*pos)[3],float(*vec)[3],float(*v_buff)[3],float(*sti),float(*e),float(*J),float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS])
{
  float xn,yn,zn,sq,dis;
  unsigned int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned int index = (blockDim.x * Grid_x) * (blockDim.y * Grid_y) * thread_idz + (blockDim.x * Grid_x) * thread_idy + thread_idx ;
  int colnum=0;
  float tmptime=0;
  int tmpindex=0;
  int coldex=0;
  float repul=0;
  float rvec[3]={0};
  float Vl[3]={0};
  float Vr[3]={0};
  float Vh[3]={0};
  float vl_buff[3]={0};
  float vr_buff[3]={0};
  float vcol_buff[3]={0};
  float dotV=0;
  xn = pos[index][X];
  yn = pos[index][Y];
  zn = pos[index][Z];
  vl_buff[X]=vec[index][X];
  vl_buff[Y]=vec[index][Y];
  vl_buff[Z]=vec[index][Z];
  for (int i = 0 ; i < NUM_POINTS; i++){
    if(colindex[index][i]!=NUM_POINTS){
      colnum++;
    }
  }
  if(colnum>0){
      //衝突域侵入からの経過時間をインデックス付きソート
      for(int i = 0 ; i < NUM_POINTS; i++){
        for(int j = i+1; j < NUM_POINTS; j++){
          if(coltime[index][i] > coltime[index][j]){
            tmptime=coltime[index][i];
            tmpindex=colindex[index][i];
            coltime[index][i]=coltime[index][j];
            colindex[index][i]=colindex[index][j];
            coltime[index][j]=tmptime;
            colindex[index][j]=tmpindex;
          }
        }
      }
      //衝突域侵入からの経過時間が長いものから処理
      for (int i=NUM_POINTS-1 ; i>=NUM_POINTS-colnum; i--){
	coldex=colindex[index][i];
	sq = (float)pow((double)(xn-pos[coldex][X]),2)+pow((double)(yn-pos[coldex][Y]),2)+pow((double)(zn-pos[coldex][Z]),2);
	dis = (float)sqrt((double)sq);
	//衝突の運動量の単位ベクトル
	rvec[X]=(pos[coldex][X]-xn)/dis;
	rvec[Y]=(pos[coldex][Y]-yn)/dis;
	rvec[Z]=(pos[coldex][Z]-zn)/dis;
	//自分の速度ベクトルの法線成分
	dotV=rvec[X]*vl_buff[X]+rvec[Y]*vl_buff[Y]+rvec[Z]*vl_buff[Z];
        Vl[X]=dotV*rvec[X];
        Vl[Y]=dotV*rvec[Y];
        Vl[Z]=dotV*rvec[Z];
	//相手の速度ベクトルの法線成分
	dotV=rvec[X]*vec[coldex][X]+rvec[Y]*vec[coldex][Y]+rvec[Z]*vec[coldex][Z];
	Vr[X]=dotV*rvec[X];
	Vr[Y]=dotV*rvec[Y];
	Vr[Z]=dotV*rvec[Z];
	//自分の速度ベクトルの水平成分
	Vh[X]=vl_buff[X]-Vl[X];
	Vh[Y]=vl_buff[Y]-Vl[Y];
	Vh[Z]=vl_buff[Z]-Vl[Z];
	//反発係数は小さいほうを優先
	repul=e[index];
	if (e[coldex] < e[index]) {
	  repul=e[coldex];
	}
	//速度更新 
        vcol_buff[X]=Vh[X]+((1+repul)*Vr[X]+(1-repul)*Vl[X])/2;
        vcol_buff[Y]=Vh[Y]+((1+repul)*Vr[Y]+(1-repul)*Vl[Y])/2;
        vcol_buff[Z]=Vh[Z]+((1+repul)*Vr[Z]+(1-repul)*Vl[Z])/2;
	//相手の速度計算
        vr_buff[X]=vec[coldex][X]-Vr[X]+((1+repul)*Vl[X]+(1-repul)*Vr[X])/2;
        vr_buff[Y]=vec[coldex][Y]-Vr[Y]+((1+repul)*Vl[Y]+(1-repul)*Vr[Y])/2;
        vr_buff[Z]=vec[coldex][Z]-Vr[Z]+((1+repul)*Vl[Z]+(1-repul)*Vr[Z])/2;
	//衝突エネルギーを粘性の比で分配し熱エネルギー変換
	double Energy=0.5*M*(pow(vec[coldex][X],2)+pow(vec[coldex][Y],2)+pow(vec[coldex][Z],2)+pow(vl_buff[X],2)+pow(vl_buff[Y],2)+pow(vl_buff[Z],2) - (pow(vcol_buff[X],2)+pow(vcol_buff[Y],2)+pow(vcol_buff[Z],2)+pow(vr_buff[X],2)+pow(vr_buff[Y],2)+pow(vr_buff[Z],2))) / pow(scale,2) * 1000000;
	J[index] += Energy / (pow(10.0,(double)(sti[index]-sti[coldex]))+1);
	//温度上限10000000度とする
	if (J[index] > M * cap * 10000000){
	  J[index] = M * cap * 10000000;
	}
        vl_buff[X]=vcol_buff[X];
        vl_buff[Y]=vcol_buff[Y];
        vl_buff[Z]=vcol_buff[Z];
	//粘性と反発係数の更新 反発係数は温度上昇に対し線形に降下、粘性は100度上昇で1桁降下
	e[index] = 1 - ((1-ref)/temp * J[index]/M/cap);
	if ( e[index] < 0 ){ e[index] = 0; }
	if ( e[index] > 1 ){ e[index] = 1; }
	sti[index] = visc - ((J[index]/M/cap - temp) / 100);   
      }
      v_buff[index][X]=vl_buff[X];
      v_buff[index][Y]=vl_buff[Y];
      v_buff[index][Z]=vl_buff[Z];
  }
  //放射冷却
  J[index]-=cool*(J[index]/M/cap)*(J[index]/M/cap)*(J[index]/M/cap)*(J[index]/M/cap)*ANIM;
  //絶対零度以下にはならない
  if (J[index] < 0) {
    J[index] = 0;
  }
}
//重力影響後の速度を計算
__global__ void grav_v(float(*pos)[3],float(*vec)[3],float(*v_buff)[3],int(*colindex)[NUM_POINTS])
{
  float xn,yn,zn,vx,vy,vz,sq,dis;
  unsigned int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned int index = (blockDim.x * Grid_x) * (blockDim.y * Grid_y) * thread_idz + (blockDim.x * Grid_x) * thread_idy + thread_idx ;
  int colnum=0;
  float gravity=0;
  xn = pos[index][X];
  yn = pos[index][Y];
  zn = pos[index][Z];
  for (int i = 0 ; i < NUM_POINTS; i++){
    if(colindex[index][i]!=NUM_POINTS){
      colnum++;
    }
  }
  if(colnum==0){
    //衝突なしなら自分以外のすべてから重力影響を受ける
    vx = vec[index][X];
    vy = vec[index][Y];
    vz = vec[index][Z]; 
    for (int i = 0 ; i < NUM_POINTS; i++){
      if (i!=index) {
	sq = (float)pow((double)(xn-pos[i][X]),2) + pow((double)(yn-pos[i][Y]),2) + pow((double)(zn-pos[i][Z]),2);
	gravity=GRAV*M/sq*scale*scale;
	dis = (float)sqrt((double)sq);
	vx += ((pos[i][X]-xn)/dis)*gravity*ANIM*scale;
	vy += ((pos[i][Y]-yn)/dis)*gravity*ANIM*scale;
	vz += ((pos[i][Z]-zn)/dis)*gravity*ANIM*scale;
      }
    }
  }
  
  else if(colnum <= 12){//六方最密充填
    //衝突ありなら自分と衝突対象以外から重力影響を受ける 
    vx = v_buff[index][X];
    vy = v_buff[index][Y];
    vz = v_buff[index][Z];    
    for (int i = 0 ; i < NUM_POINTS; i++){
      sq = (float)pow((double)(xn-pos[i][X]),2) + pow((double)(yn-pos[i][Y]),2) + pow((double)(zn-pos[i][Z]),2);
      gravity=GRAV*M/sq*scale*scale;
      dis = (float)sqrt((double)sq);
      if(dis > 2 * R) {
	vx += ((pos[i][X]-xn)/dis)*gravity*ANIM*scale;
	vy += ((pos[i][Y]-yn)/dis)*gravity*ANIM*scale;
	vz += ((pos[i][Z]-zn)/dis)*gravity*ANIM*scale;
      }
    }    
  }
  else{
    vx = v_buff[index][X];
    vy = v_buff[index][Y];
    vz = v_buff[index][Z];
  }
  
  v_buff[index][X] = vx;
  v_buff[index][Y] = vy;
  v_buff[index][Z] = vz;
}
__global__ void grav_vupdate(float(*vec)[3],float(*v_buff)[3])
{
  unsigned int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned int index = (blockDim.x * Grid_x) * (blockDim.y * Grid_y) * thread_idz + (blockDim.x * Grid_x) * thread_idy + thread_idx ;
  vec[index][X]=v_buff[index][X];
  vec[index][Y]=v_buff[index][Y];
  vec[index][Z]=v_buff[index][Z];
}
//バッファ類クリア
__global__ void buff_clear(float(*v_buff)[3],float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS],float(*p_buff)[3])
{
  unsigned int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned int index = (blockDim.x * Grid_x) * (blockDim.y * Grid_y) * thread_idz + (blockDim.x * Grid_x) * thread_idy + thread_idx ;
  for (int i=0; i < 3; i++){
    v_buff[index][i]=0;
    p_buff[index][i]=0;
  }
  for (int i=0; i < NUM_POINTS; i++){
    coltime[index][i]=0;
    colindex[index][i]=NUM_POINTS;
  }
}
//重力影響後の座標を決定
__global__ void grav_p(float(*pos)[3], float(*vec)[3])
{
  float xn,yn,zn,vx,vy,vz;
  unsigned int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned int index = ( blockDim.x * (Grid_x - 1) + blockDim.x ) * ( blockDim.y * (Grid_y - 1) + blockDim.y ) * thread_idz + ( blockDim.x * (Grid_x - 1) + blockDim.x ) * thread_idy + thread_idx ;
  xn = pos[index][X];
  yn = pos[index][Y];
  zn = pos[index][Z];
  vx = vec[index][X];
  vy = vec[index][Y];
  vz = vec[index][Z];
  pos[index][X] = xn + vx * ANIM;
  pos[index][Y] = yn + vy * ANIM;
  pos[index][Z] = zn + vz * ANIM;
}

// 粒子を初期位置に配置．
void setInitialPosition(void)
{
  for (int i = 0; i < NUM_POINTS; i++) {
    
    for (int j = 0 ; j < 3 ; j++){
      h_point[i][j] = (float)(rand()-rand()) / RAND_MAX * INIT_WIDTH/vision*2 ;
      v_point[i][j] = 0;                                                                                                   
      hv_buff[i][j] = 0;
      hp_buff[i][j] = 0;
    }
    e_point[i]=ref;
    J_point[i]=cap*M*temp;

    /*
    int earth_points = NUM_POINTS - (NUM_POINTS/64);
    if(i < earth_points){
      for (int j = 0 ; j < 3 ; j++){
	h_point[i][j] = (float)(rand()-rand()) / RAND_MAX * INIT_WIDTH/vision/2 ;
	v_point[i][j] = 0;
	hv_buff[i][j] = 0;
      }
      e_point[i]=ref;
      J_point[i]=cap*M*temp;
    }
    else {
      h_point[i][X] = (float)(rand()-rand()) / RAND_MAX * INIT_WIDTH/vision/6 + MOONOFFSET_X;
      h_point[i][Y] = (float)(rand()-rand()) / RAND_MAX * INIT_WIDTH/vision/6 + MOONOFFSET_Y;
      h_point[i][Z] = (float)(rand()-rand()) / RAND_MAX * INIT_WIDTH/vision/6 + MOONOFFSET_Z;
      v_point[i][X] = -(MOONOFFSET_X*scale/ANIM)/4.5;
      v_point[i][Y] = -(MOONOFFSET_Y*scale/ANIM)/5;
      v_point[i][Z] = -(MOONOFFSET_Z*scale/ANIM)/5;
      for (int j = 0 ; j < 3 ; j++){
	hv_buff[i][j] = 0;
      }
      e_point[i]=0;
      J_point[i]=cap*M*temp*10;
    }
    */
    st_point[i]=visc;
    for (int j = 0; j < NUM_POINTS; j++) {
      colsyncindex[i][j]=NUM_POINTS;
    }
  }
  checkCudaErrors(cudaMalloc((void**)&d_point, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dv_point, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&v_buff, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&p_buff, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dst_point, NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&de_point, NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dJ_point, NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dcolsynctime, NUM_POINTS*NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dcolsyncindex, NUM_POINTS*NUM_POINTS * sizeof(int)));
  checkCudaErrors(cudaMemcpy(d_point, h_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dv_point, v_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(v_buff, hv_buff, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(p_buff, hp_buff, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dst_point, st_point, NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(de_point, e_point, NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dJ_point, J_point, NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcolsynctime, colsynctime, NUM_POINTS*NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcolsyncindex, colsyncindex, NUM_POINTS*NUM_POINTS * sizeof(int) , cudaMemcpyHostToDevice));
}
//CUDA実行関数
void launchGPUKernel(unsigned int num_particles,float(*pos)[3],float(*vec)[3],float(*v_buff)[3],float(*sti),float(*e),float(*J),float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS],float(*p_buff)[3])
{
    dim3 grid(Grid_x,Grid_y,Grid_z);
    dim3 block(Block_x,Block_y,Block_z);
    grav_coldetect<<<grid , block>>>(pos, vec,coltime,colindex,p_buff);
    grav_padjust<<<grid , block>>>(pos, p_buff);
    grav_colv<<<grid , block>>>(pos,vec,v_buff,sti,e,J,coltime,colindex);
    grav_v<<<grid , block>>>(pos,vec,v_buff,colindex);
    grav_vupdate<<<grid , block>>>(vec,v_buff);
    buff_clear<<<grid , block>>>(v_buff,coltime,colindex,p_buff);
    grav_p<<<grid , block>>>(pos,vec);
}
//アニメーション動作
void runGPUKernel(void)
{
  launchGPUKernel(NUM_POINTS, d_point, dv_point,v_buff,dst_point, de_point,dJ_point,dcolsynctime,dcolsyncindex,p_buff);
  checkCudaErrors(cudaMemcpy(h_point, d_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(v_point, dv_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hv_buff, v_buff, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hp_buff, p_buff, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(st_point, dst_point, NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(e_point, de_point, NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(J_point, dJ_point, NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(colsynctime,dcolsynctime, NUM_POINTS*NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(colsyncindex,dcolsyncindex, NUM_POINTS*NUM_POINTS * sizeof(int) , cudaMemcpyDeviceToHost));
  anim_time += anim_dt;
}
//ビュー定義
void defineViewMatrix(double phi, double theta)
{
  unsigned int i;
  double c, s, xy_dist;
  double x_axis[3], y_axis[3], z_axis[3];

  // 視点の設定．
  eye[Z] = sin(theta * PI / 180.0);
  xy_dist = cos(theta * PI / 180.0);
  c = cos(phi * PI / 180.0);
  s = sin(phi * PI / 180.0);
  eye[X] = xy_dist * c;
  eye[Y] = xy_dist * s;
  up[X] = - c * eye[Z];
  up[Y] = - s * eye[Z];
  up[Z] = s * eye[Y] + c * eye[X];
  normVec(up);
  // 視点を原点とする座標系の定義．
  for (i = 0; i < 3; i++)
    {
      z_axis[i] = eye[i] - center[i];
    }
  normVec(z_axis);
  cross(up, z_axis, x_axis);
  normVec(x_axis);
  cross(z_axis, x_axis, y_axis);
  gluLookAt(eye[X], eye[Y], eye[Z], center[X], center[Y], center[Z], up[X], up[Y], up[Z]); 
}

void metaball (float pos[3], float color[3]) {
  double margin=0;
  double view[3]={0};
  double TH=theta;
  double PH=-phi;
  for (int i = 0 ; i < dev_points ; i ++)                                                                                    
    {
      view[X] = 0;
      view[Y] = R * cos(i * PI * 2 / dev);                                                   
      view[Z] = R * sin(i * PI * 2 / dev); 
      //極座標変換
      point[i][X] = view[X] * cos(TH * PI / 180) * cos(PH * PI / 180) + view[Y] * sin(PH * PI / 180) - view[Z] * sin(TH * PI / 180) * cos(PH * PI / 180);
      point[i][Y] = - view[X] * cos(TH * PI / 180) * sin(PH * PI / 180) + view[Y] * cos(PH * PI / 180) + view[Z] * sin(TH * PI / 180) * sin(PH * PI / 180);
      point[i][Z] = view[X] * sin(TH * PI / 180) + view[Z] * cos(TH * PI / 180);            
    }
  //中心の球体を円で描き視点に合わせて向きを変えることで球体に見せる
  glBegin(GL_TRIANGLE_FAN);
  glColor4f(1,1,1,0.3);
  glVertex3d(pos[X],pos[Y],pos[Z]);
  for (int i = 0 ; i < dev_points ; i ++)
    {
      glVertex3d(point[i][X] + pos[X], point[i][Y] + pos[Y], point[i][Z] + pos[Z]);
    }
  glEnd(); 
  //周囲のボヤ
  int mh[dev_points];
  for (int i = 0 ; i < dev_points ; i ++)
    {
      mh[i]=1;
    }

  glBegin(GL_POINTS);
  glColor4f(color[X],color[Y],color[Z],0.1);
  for (int k = 0; k < hollow; k++) {
    margin=0.5/hollow*k+1;
    for (int i = 0 ; i < dev_points ; i ++)
      {
        if((mh[i]==1 || mh[i-1]==1 || mh[i+1]==1) && (rand() % dev) < (dev * (hollow-k/2)/hollow))
          glVertex3d(margin*point[i][X] + pos[X], margin*point[i][Y] + pos[Y], margin*point[i][Z] + pos[Z]);
        else
          mh[i]=0;
      }
  }
  glEnd();
}
void display(void)
{
    light_pos[0] = (float)eye[X];
    light_pos[1] = (float)eye[Y];
    light_pos[2] = (float)eye[Z];
    light_pos[3] = 0.0f;
    //CUDA開始
    runGPUKernel();
    // 光源の設定
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-vision_size-right_motion/2, vision_size+right_motion/2, -vision_size-right_motion/2, vision_size+right_motion/2, -100*vision_size, 100*vision_size);
    glViewport(0, 0, window_width, window_height);
    defineViewMatrix(phi, theta);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //メタボール
    float color[3]={0};
    for (int k = 0 ; k < NUM_POINTS ; k++)
      {
	//温度によって色を変化
	if(J_point[k]/M/cap-temp < resol){ 
          color[X]=1.0;                                                                                                     
          color[Y]=1.0;                                                                                                      
          color[Z]=1.0-(J_point[k]/M/cap-temp)/resol;
        }
	else if(J_point[k]/M/cap-temp < 2 * resol){
          color[X]=1.0;
          color[Y]=1.0-(J_point[k]/M/cap-temp-resol)/resol;
          color[Z]=0.0;
        }
        else {
          color[X]=1.0;
          color[Y]=0.0;
          color[Z]=0.0;
        }
	metaball(h_point[k],color);
      }
    glutSwapBuffers();
    glutPostRedisplay();
}

void mouse_button(int button, int state, int x, int y)
{
  if ((state == GLUT_DOWN) && (button == GLUT_LEFT_BUTTON))
    motion_p = true;
  if ((state == GLUT_DOWN) && (button == GLUT_RIGHT_BUTTON))
    motion_w = true;
  else if (state == GLUT_UP) {
    motion_p = false;
    motion_w = false;
  }
  mouse_old_x = x;
  mouse_old_y = y;
}
void mouse_motion(int x, int y)
{
  int dx, dy;
  dx = x - mouse_old_x;
  dy = y - mouse_old_y;
  if (motion_p) {
    phi -= dx * 0.2;
    theta += dy * 0.2;
  }
  if (motion_w) {
    right_motion += dx / 10;
    up_motion -= dy / 10;
  }
  mouse_old_x = x;
  mouse_old_y = y;
  glutPostRedisplay();
}

void resize(int width, int height)
{
  window_width = width;
  window_height = height;
}
void keyboard(unsigned char key, int x, int y)
{
  switch (key) {
  case 'q':
  case 'Q':
  case '\033':
    exit(0);
    
  default:
    break;
  }
}

bool initGL(void)
{
  glClearColor(0.0f, 0.0f , 0.0f, 0.5f);
  glEnable(GL_DEPTH_TEST);
  glClearDepth(1.0);
  glDepthFunc(GL_LESS);
  glEnable(GL_LIGHT0);
  return true;
}

int main(int argc, char** argv)
{
  point = (double **)malloc(sizeof(double *) * dev_points);
  for (int i = 0 ; i < dev_points ; i++)
    {
      point[i] = (double *)malloc(sizeof(double) * 3);
    } 
  glutInit(&argc, argv);
  //glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE); 
  glutInitWindowSize(window_width, window_height); 
  glutCreateWindow("3D CUDA Simulation");
  glutDisplayFunc(display);
  glutReshapeFunc(resize);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse_button);
  glutMotionFunc(mouse_motion);
  setInitialPosition();
  if (!initGL())
    return 1;
  glutMainLoop();
  
  cudaFree(dst_point);
  cudaFree(de_point);
  cudaFree(dJ_point);
  for (int i = 0 ; i < dev_points ; i++)
    {
      free (point[i]);
      cudaFree(d_point[i]);
      cudaFree(dv_point[i]);
      cudaFree(v_buff[i]);
      cudaFree(p_buff[i]);
      cudaFree(dcolsynctime[i]);
      cudaFree(dcolsyncindex[i]);
    }
  free (point);
  cudaFree(d_point);
  cudaFree(dv_point);
  cudaFree(v_buff);
  cudaFree(p_buff);
  cudaFree(dcolsynctime);
  cudaFree(dcolsyncindex);
  cudaDeviceReset();
  return 0;
}
