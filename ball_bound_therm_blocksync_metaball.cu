#include <stdio.h>
#include <stdlib.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_runtime.h>

//TDB
//放射冷却
//メタボール
//VBO
//初期座標を球形にする
//衝突角度や距離をスライドで変更できるようにする

#define PI 3.141592653589793
//物理パラメータ
#define cap 1000
#define ref 0.6
#define temp 4273
#define visc 9
#define GRAV (6.674*0.00000000000000000001)
#define density (2.5 * 1000000000000)
//粒子形状
#define rad 40 //km
#define M (4 / 3 * PI * rad*rad*rad* density)//kg
//描写設定
#define dev 30
#define resol 30
#define X 0
#define Y 1
#define Z 2
#define ANIM_START 0
#define ANIM 500
#define scale 0.01
#define colmargin 1.1
#define R (rad * scale)
#define INIT_WIDTH 800
#define INIT_HEIGHT 800
#define vision 40
#define Grid_x 16//block間は__syncthreadでは同期不可
#define Grid_y 8
#define Grid_z 1
#define Block_x 4
#define Block_y 2
#define Block_z 1

#define NUM_POINTS (Grid_x*Grid_y*Grid_z*Block_x*Block_y*Block_z)

unsigned int num_points = (dev + 1) * (dev + 1);
unsigned int window_width = INIT_WIDTH;
unsigned int window_height = INIT_HEIGHT;
double vision_size = vision;
float right_motion=0;
float up_motion=0;
double left, right, bottom, top;
float h_point[NUM_POINTS][3];
float v_point[NUM_POINTS][3];
float st_point[NUM_POINTS];
float e_point[NUM_POINTS];
float J_point[NUM_POINTS];
float h_buff[NUM_POINTS][3]={0};
float anim_time = ANIM_START;
float anim_dt = ANIM;
double phi = 30.0;
double theta = 30.0;
float light_pos[4];
int mouse_old_x, mouse_old_y;
bool motion_p;
bool motion_w;
double eye[3];
double center[3] = {0.0, 0.0, 0.0};
double up[3];
double ** point;
float (*d_point)[3];
float (*dv_point)[3];
float (*dst_point);
float (*de_point);
float (*dJ_point);
float (*v_buff)[3];
float colsynctime[NUM_POINTS][NUM_POINTS]={0};
int colsyncindex[NUM_POINTS][NUM_POINTS]={0};
float (*dcolsynctime)[NUM_POINTS];
int (*dcolsyncindex)[NUM_POINTS];
__global__ void grav_coldetect(float(*pos)[3],float(*vec)[3],float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS]);
__global__ void grav_colv(float(*pos)[3],float(*vec)[3],float(*v_buff)[3],float(*sti),float(*e),float(*J),float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS]);
__global__ void grav_v(float(*pos)[3],float(*vec)[3],float(*v_buff)[3],int(*colindex)[NUM_POINTS]);
__global__ void grav_vupdate(float(*vec)[3],float(*v_buff)[3]);
__global__ void buff_clear(float(*v_buff)[3],float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS]);
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
__global__ void grav_coldetect(float(*pos)[3],float(*vec)[3],float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS])
{
  float xn,yn,zn,vx,vy,vz,dis,sq;
  unsigned int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned int index = (blockDim.x * Grid_x) * (blockDim.y * Grid_y) * thread_idz + (blockDim.x * Grid_x) * thread_idy + thread_idx ;
  float rvec[3]={0};
  xn = pos[index][0];
  yn = pos[index][1];
  zn = pos[index][2];
  vx = vec[index][0];
  vy = vec[index][1];
  vz = vec[index][2];
  for (int i = 0 ; i < NUM_POINTS; i++)
    {
      sq = (float)pow((double)(xn-pos[i][0]),2) + pow((double)(yn-pos[i][1]),2) + pow((double)(zn-pos[i][2]),2);
      dis = (float)sqrt((double)sq);
      rvec[0]=(pos[i][0]-xn)/dis;
      rvec[1]=(pos[i][1]-yn)/dis;
      rvec[2]=(pos[i][2]-zn)/dis;
      //衝突域侵入判定
      if (dis > 2 * R * colmargin  && i != index)
	{
	  colindex[index][i]=NUM_POINTS;
	}
      else if (dis <= 2 * R * colmargin && i != index)
	{
	  //衝突域侵入からの経過の時間を記録　
	  colindex[index][i]=i;
	  coltime[index][i]=(2*R*colmargin-dis)/((vx-vec[i][0])*rvec[0]+(vy-vec[i][1])*rvec[1]+(vz-vec[i][2])*rvec[2]);
	}
      else
	{
	  colindex[index][i]=NUM_POINTS;
	}
    }
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
  xn = pos[index][0];
  yn = pos[index][1];
  zn = pos[index][2];
  vl_buff[0]=vec[index][0];
  vl_buff[1]=vec[index][1];
  vl_buff[2]=vec[index][2];
  for (int i = 0 ; i < NUM_POINTS; i++){
    if(colindex[index][i]!=NUM_POINTS){
      colnum++;
    }
  }
  if(colnum>0){
      //衝突域侵入からの経過時間をインデックス付きソート
      for (int i = 0 ; i < NUM_POINTS; i++){
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
	sq = (float)pow((double)(xn-pos[coldex][0]),2)+pow((double)(yn-pos[coldex][1]),2)+pow((double)(zn-pos[coldex][2]),2);
	dis = (float)sqrt((double)sq);
	//衝突の運動量の単位ベクトル
	rvec[0]=(pos[coldex][0]-xn)/dis;
	rvec[1]=(pos[coldex][1]-yn)/dis;
	rvec[2]=(pos[coldex][2]-zn)/dis;
	//自分の速度ベクトルの法線成分
	dotV=rvec[0]*vl_buff[0]+rvec[1]*vl_buff[1]+rvec[2]*vl_buff[2];
        Vl[0]=dotV*rvec[0];
        Vl[1]=dotV*rvec[1];
        Vl[2]=dotV*rvec[2];
	//相手の速度ベクトルの法線成分
	dotV=rvec[0]*vec[coldex][0]+rvec[1]*vec[coldex][1]+rvec[2]*vec[coldex][2];
	Vr[0]=dotV*rvec[0];
	Vr[1]=dotV*rvec[1];
	Vr[2]=dotV*rvec[2];
	//自分の速度ベクトルの水平成分
	Vh[0]=vl_buff[0]-Vl[0];
	Vh[1]=vl_buff[1]-Vl[1];
	Vh[2]=vl_buff[2]-Vl[2];
	//反発係数は小さいほうを優先
	repul=e[index];
	if (e[coldex] < e[index]) {
	  repul=e[coldex];
	}
	//速度更新 
        vcol_buff[0]=Vh[0]+((1+repul)*Vr[0]+(1-repul)*Vl[0])/2;
        vcol_buff[1]=Vh[1]+((1+repul)*Vr[1]+(1-repul)*Vl[1])/2;
        vcol_buff[2]=Vh[2]+((1+repul)*Vr[2]+(1-repul)*Vl[2])/2;
	//相手の速度計算
        vr_buff[0]=vec[coldex][0]-Vr[0]+((1+repul)*Vl[0]+(1-repul)*Vr[0])/2;
        vr_buff[1]=vec[coldex][1]-Vr[1]+((1+repul)*Vl[1]+(1-repul)*Vr[1])/2;
        vr_buff[2]=vec[coldex][2]-Vr[2]+((1+repul)*Vl[2]+(1-repul)*Vr[2])/2;
	//衝突エネルギーを粘性の比で分配し熱エネルギー変換 TBD 放射冷却
	double Energy=0.5*M*(pow(vec[coldex][0],2)+pow(vec[coldex][1],2)+pow(vec[coldex][2],2)+pow(vl_buff[0],2)+pow(vl_buff[1],2)+pow(vl_buff[2],2) - (pow(vcol_buff[0],2)+pow(vcol_buff[1],2)+pow(vcol_buff[2],2)+pow(vr_buff[0],2)+pow(vr_buff[1],2)+pow(vr_buff[2],2))) / pow(scale,2) * 1000000;
	J[index] += Energy / (pow(10.0,(double)(sti[index]-sti[coldex]))+1);
	//温度上限15000度とする
	if (J[index] > M * cap * 15000){
	  J[index] = M * cap * 15000;
	}
        vl_buff[0]=vcol_buff[0];
        vl_buff[1]=vcol_buff[1];
        vl_buff[2]=vcol_buff[2];
	//粘性と反発係数の更新 反発係数は温度上昇に対し線形に降下、粘性は100度上昇で1桁降下
	e[index] = 1 - ((1-ref)/temp * J[index]/M/cap);
	if ( e[index] < 0 ){ e[index] = 0; }
	if ( e[index] > 1 ){ e[index] = 1; }
	sti[index] = visc - ((J[index]/M/cap - temp) / 100);   
      }
      v_buff[index][0]=vl_buff[0];
      v_buff[index][1]=vl_buff[1];
      v_buff[index][2]=vl_buff[2];
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
  xn = pos[index][0];
  yn = pos[index][1];
  zn = pos[index][2];
  for (int i = 0 ; i < NUM_POINTS; i++){
    if(colindex[index][i]!=NUM_POINTS){
      colnum++;
    }
  }
  if(colnum==0){
    //衝突なしなら自分以外のすべてから重力影響を受ける
    vx = vec[index][0];
    vy = vec[index][1];
    vz = vec[index][2]; 
    for (int i = 0 ; i < NUM_POINTS; i++){
      if (i!=index) {
	sq = (float)pow((double)(xn-pos[i][0]),2) + pow((double)(yn-pos[i][1]),2) + pow((double)(zn-pos[i][2]),2);
	gravity=GRAV*M/sq*scale*scale;
	dis = (float)sqrt((double)sq);
	vx += ((pos[i][0]-xn)/dis)*gravity*ANIM*scale;
	vy += ((pos[i][1]-yn)/dis)*gravity*ANIM*scale;
	vz += ((pos[i][2]-zn)/dis)*gravity*ANIM*scale;
      }
    }
  }
  else {
    //衝突ありなら自分と衝突対象以外から重力影響を受ける 
    vx = v_buff[index][0];
    vy = v_buff[index][1];
    vz = v_buff[index][2];
  }
  v_buff[index][0] = vx;
  v_buff[index][1] = vy;
  v_buff[index][2] = vz;
}
__global__ void grav_vupdate(float(*vec)[3],float(*v_buff)[3])
{
  unsigned int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned int index = (blockDim.x * Grid_x) * (blockDim.y * Grid_y) * thread_idz + (blockDim.x * Grid_x) * thread_idy + thread_idx ;
  vec[index][0]=v_buff[index][0];
  vec[index][1]=v_buff[index][1];
  vec[index][2]=v_buff[index][2];
}
//バッファ類クリア
__global__ void buff_clear(float(*v_buff)[3],float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS])
{
  unsigned int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned int index = (blockDim.x * Grid_x) * (blockDim.y * Grid_y) * thread_idz + (blockDim.x * Grid_x) * thread_idy + thread_idx ;
  for (int i=0; i < 3; i++){
    v_buff[index][i]=0;
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
  xn = pos[index][0];
  yn = pos[index][1];
  zn = pos[index][2];
  vx = vec[index][0];
  vy = vec[index][1];
  vz = vec[index][2];
  pos[index][0] = xn + vx * ANIM;
  pos[index][1] = yn + vy * ANIM;
  pos[index][2] = zn + vz * ANIM;
}

// 粒子を初期位置に配置．
void setInitialPosition(void)
{
  for (int i = 0; i < NUM_POINTS; i++) {
    for (int j = 0 ; j < 3 ; j++){
      h_point[i][j] = (float)(rand()-rand()) / RAND_MAX * INIT_WIDTH/vision*2 ;
      v_point[i][j] = 0;
      h_buff[i][j] = 0;
    }
    st_point[i]=visc;
    e_point[i]=ref;
    J_point[i]=cap*M*temp;
    for (int j = 0; j < NUM_POINTS; j++) {
      colsyncindex[i][j]=NUM_POINTS;
    }
  }
  checkCudaErrors(cudaMalloc((void**)&d_point, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dv_point, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&v_buff, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dst_point, NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&de_point, NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dJ_point, NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dcolsynctime, NUM_POINTS*NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dcolsyncindex, NUM_POINTS*NUM_POINTS * sizeof(int)));
  checkCudaErrors(cudaMemcpy(d_point, h_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dv_point, v_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(v_buff, h_buff, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dst_point, st_point, NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(de_point, e_point, NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dJ_point, J_point, NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcolsynctime, colsynctime, NUM_POINTS*NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcolsyncindex, colsyncindex, NUM_POINTS*NUM_POINTS * sizeof(int) , cudaMemcpyHostToDevice));
}
//CUDA実行関数
void launchGPUKernel(unsigned int num_particles,float(*pos)[3],float(*vec)[3],float(*v_buff)[3],float(*sti),float(*e),float(*J),float(*coltime)[NUM_POINTS],int(*colindex)[NUM_POINTS])
{
    dim3 grid(Grid_x,Grid_y,Grid_z);
    dim3 block(Block_x,Block_y,Block_z);
    grav_coldetect<<<grid , block>>>(pos, vec,coltime,colindex);
    grav_colv<<<grid , block>>>(pos,vec,v_buff,sti,e,J,coltime,colindex);
    grav_v<<<grid , block>>>(pos,vec,v_buff,colindex);
    grav_vupdate<<<grid , block>>>(vec,v_buff);
    buff_clear<<<grid , block>>>(v_buff,coltime,colindex);
    grav_p<<<grid , block>>>(pos,vec);
}
//アニメーション動作
void runGPUKernel(void)
{
  launchGPUKernel(NUM_POINTS, d_point, dv_point,v_buff,dst_point, de_point,dJ_point,dcolsynctime,dcolsyncindex);
  checkCudaErrors(cudaMemcpy(h_point, d_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(v_point, dv_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_buff, v_buff, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
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
  double nrml_vec[3];
  double margin=0;
  double view[3]={0};
  double TH=theta;
  double PH=-phi;
  for (int i = 0 ; i < dev + 1; i ++)
    {
      for (int j = 0 ; j < dev + 1; j++)
        {
	  view[0] = R * cos(j * PI * 2 / dev) * cos(i * PI * 2 / dev);
	  view[1] = R * sin(j * PI * 2 / dev) * cos(i * PI * 2 / dev);
          view[2] = R * sin(i * PI * 2 / dev);
	  //極座標変換
	  point[i * dev + j][X] = view[0] * cos(TH * PI / 180) * cos(PH * PI / 180) + view[1] * sin(PH * PI / 180) - view[2] * sin(TH * PI / 180) * cos(PH * PI / 180);
	  point[i * dev + j][Y] = - view[0] * cos(TH * PI / 180) * sin(PH * PI / 180) + view[1] * cos(PH * PI / 180) + view[2] * sin(TH * PI / 180) * sin(PH * PI / 180);
	  point[i * dev + j][Z] = view[0] * sin(TH * PI / 180) + view[2] * cos(TH * PI / 180);

	}
    }
  //中心の球体 
  glBegin(GL_QUADS);
  glColor4f(1,1,1,0.3);
  for (int i = 0 ; i < dev + 1 ; i ++)
    {
      for (int j = 0 ; j < 2 * dev + 1 ; j++)
	{
	  normal(point[i * (dev-1) + j],point[(i + 1) * (dev-1) + j + 1],point[(i+1) * (dev-1) + j],nrml_vec);
	  glNormal3dv(nrml_vec);
	  glVertex3d(point[i * (dev-1) + j][X] + pos[X], point[i * (dev-1) + j][Y] + pos[Y], point[i * (dev-1) + j][Z] + pos[Z]);
	  glVertex3d(point[(i + 1) * (dev-1) + j][X] + pos[X],point[(i + 1) * (dev-1) + j][Y] + pos[Y],point[(i + 1) * (dev-1) + j][Z] + pos[Z]);
	  glVertex3d(point[(i + 1) * (dev-1) + j + 1][X] + pos[X], point[(i + 1) * (dev-1) + j + 1][Y] + pos[Y], point[(i + 1) * (dev-1) + j + 1][Z] + pos[Z]);
	  glVertex3d(point[i * (dev-1) + j + 1][X] + pos[X],point[i * (dev-1) + j + 1][Y] + pos[Y],point[i * (dev-1) + j + 1][Z] + pos[Z]);
	}
    }
  glEnd(); 
  //周囲のボヤ
  glBegin(GL_POINTS);
  glColor4f(color[0],color[1],color[2],0.1);
  for (int k = 0; k < 5; k++) {
    margin=(colmargin-1)*5/(k*2)+1;
    for (int i = 0 ; i < dev + 1 ; i ++)
      {
	for (int j = 0 ; j < 2 * dev + 1 ; j++)
	  {
	    normal(point[i * (dev-1) + j],point[(i + 1) * (dev-1) + j + 1],point[(i+1) * (dev-1) + j],nrml_vec);
	    glNormal3dv(nrml_vec);
	    glVertex3d(margin*point[i * (dev-1) + j][X] + pos[X], margin*point[i * (dev-1) + j][Y] + pos[Y], margin*point[i * (dev-1) + j][Z] + pos[Z]);
	  }
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
          color[0]=1.0;                                                                                                     
          color[1]=1.0;                                                                                                      
          color[2]=1.0-(J_point[k]/M/cap-temp)/resol;
        }
	else if(J_point[k]/M/cap-temp < 2 * resol){
          color[0]=1.0;
          color[1]=1.0-(J_point[k]/M/cap-temp-resol)/resol;
          color[2]=0.0;
        }
        else {
          color[0]=1.0;
          color[1]=0.0;
          color[2]=0.0;
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
  point = (double **)malloc(sizeof(double *) * num_points);
  for (int i = 0 ; i < num_points ; i++)
    {
      point[i] = (double *)malloc(sizeof(double) * 3);
    } 
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
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
    for (int i = 0 ; i < num_points ; i++)
      {
       	free (point[i]);
	cudaFree(d_point[i]);
        cudaFree(dv_point[i]);
        cudaFree(v_buff[i]);
	cudaFree(dcolsynctime[i]);
	cudaFree(dcolsyncindex[i]);
      }
    free (point);
    cudaFree(d_point);
    cudaFree(dv_point);
    cudaFree(v_buff);
    cudaFree(dcolsynctime);
    cudaFree(dcolsyncindex);
    cudaDeviceReset();
    return 0;
}
