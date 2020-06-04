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
#include <unistd.h>

#define PI 3.141592653589793

#define cap 1000
#define ref 0.9
#define temp 4273
#define visc 21
#define GRAV (6.674*0.00000000000000000001)
#define density (2.5 * 1000000000000)

#define rad 50
#define dev 12
#define M (4 / 3 * PI * rad*rad*rad* density)

#define X 0
#define Y 1
#define Z 2
#define ANIM 100000
#define scale 0.01
#define colmargin 1.05
#define R (rad * scale)
#define INIT_WIDTH 800
#define INIT_HEIGHT 800
#define vision 40
#define Grid_x 1 //block間はsyncthreadで同期できない
#define Grid_y 1
#define Grid_z 1
#define Block_x 16
#define Block_y 4
#define Block_z 1

#define NUM_POINTS (Grid_x*Grid_y*Grid_z*Block_x*Block_y*Block_z)

unsigned int num_points = (dev + 1) * (dev + 1);
unsigned int window_width = 500;
unsigned int window_height = 500;
double init_left = -10000;
double init_right = 10000;
double init_bottom = -10000;
double init_top = 10000;
double left, right, bottom, top;
float h_point[NUM_POINTS][3];
float v_point[NUM_POINTS][3]={0};
float st_point[NUM_POINTS]={visc};
float e_point[NUM_POINTS]={ref};
float T_point[NUM_POINTS]={temp};
float J_point[NUM_POINTS]={cap*M*temp};
float anim_time = ANIM;
float anim_dt = 0.1;
double phi = 30.0;
double theta = 30.0;
float light_pos[4];
int mouse_old_x, mouse_old_y;
bool motion_p;
double eye[3];
double center[3] = {0.0, 0.0, 0.0};
double up[3];
double ** point;
float (*d_point)[3];
float (*dv_point)[3];
float (*dst_point);
float (*de_point);
float (*dT_point);
float (*dJ_point);

float colsynctime[NUM_POINTS][NUM_POINTS]={0.0};
float colsyncindex[NUM_POINTS][NUM_POINTS]={0.0};
float (*dcolsynctime)[NUM_POINTS];
float (*dcolsyncindex)[NUM_POINTS];

__global__ void grav_v(float (*pos)[3], float(*vec)[3] ,float(*sti),float(*e),float(*T),float(*J), float time, float dt,float(*coltime)[NUM_POINTS],float(*colindex)[NUM_POINTS]);
__global__ void grav_p(float (*pos)[3], float(*vec)[3] , float time, float dt);
/*
texture<float,2> timetocol;
texture<float,2> indextocol;
void bindTextures(float **coltime, float **colindex)
{
  cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaBindTexture(0, timetocol, coltime, desc);
  cudaBindTexture(0, indextocol,colindex, desc);
}
void unbindTextures(void)
{
  cudaUnbindTexture(timetocol);
  cudaUnbindTexture(indextocol);
}
*/

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

//重力影響後の速度を決定
__global__ void grav_v(float (*pos)[3],float(*vec)[3],float(*sti),float(*e),float(*T),float(*J), float time, float dt,float(*coltime)[NUM_POINTS],float(*colindex)[NUM_POINTS])
{
  double xn,yn,zn,vx,vy,vz,dis,sq;
  unsigned int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned int index = (blockDim.x * Grid_x) * (blockDim.y * Grid_y) * thread_idz + (blockDim.x * Grid_x) * thread_idy + thread_idx ;

  double v_buff[3]={0};
  //double coltime[NUM_POINTS][2]={0};
  int colnum=0;
  double gravity=0;
  xn = pos[index][0];
  yn = pos[index][1];
  zn = pos[index][2];
  vx = vec[index][0];
  vy = vec[index][1];
  vz = vec[index][2];
  v_buff[0]=vx;
  v_buff[1]=vy;
  v_buff[2]=vz;
  
  for (int i = 0 ; i < NUM_POINTS; i++)
    {
      sq = pow((double)(xn-pos[i][0]),2) + pow((double)(yn-pos[i][1]),2) + pow((double)(zn-pos[i][2]),2);
      gravity=GRAV*M/sq*scale*scale;
      dis = sqrt(sq);
      //衝突域侵入判定
      if (dis > 2 * R * colmargin  && i != index)
	{
	  //他粒子へ与える運動エネルギーと内部エネルギーの交換 
	  //J[index]-=0.5*M*fabs((float)(pow((double)vec[i][0]/scale,2)+pow((double)vec[i][1]/scale,2)+pow((double)vec[i][2]/scale,2))-(float)(pow((double)(vec[i][0]/scale + 0.5*((pos[i][0]-xn)/dis)*gravity*ANIM*ANIM),2)+pow((double)(vec[i][1]/scale + 0.5*((pos[i][1]-yn)/dis)*gravity*ANIM*ANIM),2)+pow((double)(vec[i][2]/scale + 0.5*((pos[i][2]-zn)/dis)*gravity*ANIM*ANIM),2)));
	  //速度更新
	  vx = vx + ((pos[i][0]-xn)/dis)*gravity*ANIM*scale;
	  vy = vy + ((pos[i][1]-yn)/dis)*gravity*ANIM*scale;
	  vz = vz + ((pos[i][2]-zn)/dis)*gravity*ANIM*scale;
	  colindex[index][i]=NUM_POINTS;
	}
      else if (i != index)
	{
	  //衝突域侵入からの経過の時間を記録　TBD　法線方向に直す
	  colindex[index][i]=i;
	  coltime[index][i]=(2*R*colmargin - dis)/(pow((double)(vx-vec[i][0]),2)+pow((double)(vy-vec[i][1]),2)+pow((double)(vz-vec[i][2]),2));
	  colnum++;
	}
      else
	{
          colindex[index][i]=NUM_POINTS;
	}
    }
  //__syncthreads();
  /*
  int syncchk=0;
  for (int k=0; k<7000; k++){
    for(int i=0;i<NUM_POINTS;i++){
      for(int j=0;j<NUM_POINTS;j++){
	if(colindex[i][j]==0){
	  syncchk=0;
	  break;
	}
	else{
	  syncchk=1;
	}
      }
      if(syncchk==0){
	break;
      }
    }
    if(syncchk==0){
      usleep(10);
    }
    else {
      break;
    }
  }
  */
  if(colnum>0)
    {
      //衝突域侵入からの経過時間をインデックス付きソート
      double tmp[2]={0};
      for (int i = 0 ; i < NUM_POINTS; i++){
	for(int j = i+1; j < NUM_POINTS; j++){
	  if(coltime[index][i] > coltime[index][j]){
	    tmp[0]=coltime[index][i];
	    tmp[1]=colindex[index][i];
	    coltime[index][i]=coltime[index][j];
	    colindex[index][i]=colindex[index][j];
	    coltime[index][j]=tmp[0];
	    colindex[index][j]=tmp[1];
	  }
	}
      }
      //衝突域侵入からの経過時間が長いものから処理
      for (int i=NUM_POINTS-1 ; i>=NUM_POINTS-colnum; i--){
	int coldex=coltime[i][1];
	float repul=0;
	if (coldex != index) {
	  repul=e[index];
	  //反発係数は小さいほうを優先     
	  if (e[coldex] < e[index]) {
	    repul=e[coldex];
	  }
	  //速度更新
	  v_buff[0]=(double)((1+repul)*M*vec[coldex][0]+(M-repul*M)*v_buff[0])/(M+M);
	  v_buff[1]=(double)((1+repul)*M*vec[coldex][1]+(M-repul*M)*v_buff[1])/(M+M);
	  v_buff[2]=(double)((1+repul)*M*vec[coldex][2]+(M-repul*M)*v_buff[2])/(M+M);
	  //衝突エネルギーをm^2/3*stiの比で分配し熱エネルギー変換
	  //double Energy=0.5*(1-repul*repul)*(M*(pow((double)vx/scale,2)+pow((double)vy/scale,2)+pow((double)vz/scale,2)) + M*(pow((double)vec[coldex][0]/scale,2)+pow((double)vec[coldex][1]/scale,2)+pow((double)vec[coldex][2]/scale,2)));
	  //J[index]+=Energy * pow((double)M,0.667) * pow(10.0,(double)sti[index]) / (pow((double)M,0.667) * pow(10.0,(double)sti[index]) + pow((double)M,0.667) * pow(10.0,(double)sti[coldex]));
	  //T[index]=(J[index]-0.5*M*(pow((double)v_buff[0],2)+pow((double)v_buff[1],2)+pow((double)v_buff[2],2)))/M/cap;	
	  //粘性と反発係数の更新
	  //e[index] = e[index] * (visc+(temp/100)-(T[index]/100)-0.5*log(M))/(visc-0.5*log(M));
	  //sti[index] = visc - (T[index] - temp / 100);
	}
      }
    }
  for(int i=0;i<NUM_POINTS;i++){
    coltime[index][i]=0;
    colindex[index][i]=0;
  }
  //  __syncthreads();
  /*
  for (int k=0; k<7000; k++){
    for(int i=0;i<NUM_POINTS;i++){
      for(int j=0;j<NUM_POINTS;j++){
        if(colindex[i][j]!=0){
          syncchk=1;
          break;
        }
        else{
          syncchk=0;
        }
      }
      if(syncchk==1){
        break;
      }
    }
    if(syncchk==1){
      usleep(10);
    }
    else {
      break;
    }
  }
  */
  if (colnum>0)
    {
      vec[index][0] = (float)v_buff[0];
      vec[index][1] = (float)v_buff[1];
      vec[index][2] = (float)v_buff[2];
    }
  else
    {
      vec[index][0] = (float)vx;
      vec[index][1] = (float)vy;
      vec[index][2] = (float)vz;
    }
  //内部エネルギーと運動エネルギーから熱エネルギー更新                                                                
  //T[index]=(J[index]-0.5*M*(pow((double)vec[index][0]/scale,2)+pow((double)vec[index][1]/scale,2)+pow((double)vec[index][2]/scale,2)))/M/cap;
  //粘性と反発係数の更新 
  //e[index] = e[index] * (visc+(temp/100)-(T[index]/100)-0.5*log(M))/(visc-0.5*log(M));
  //sti[index] = visc - (T[index] - temp / 100);
}

//重力影響後の座標を決定
__global__ void grav_p(float (*pos)[3], float(*vec)[3] , float time, float dt)
{
  double xn,yn,zn,vx,vy,vz;
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
  pos[index][0] = xn + vx * dt;
  pos[index][1] = yn + vy * dt;
  pos[index][2] = zn + vz * dt;
}

// 粒子を初期位置に配置．
void setInitialPosition(void)
{
  srand(12131);
  for (int i = 0; i < NUM_POINTS; i++) {
    if(rand()%2==1)
      {
	h_point[i][0] = (double)rand() / RAND_MAX * INIT_WIDTH / 10;
        h_point[i][1] = (double)rand() / RAND_MAX * INIT_WIDTH / 10;
        h_point[i][2] = (double)rand() / RAND_MAX * INIT_WIDTH / 10;
      }
    else
      {
	h_point[i][0] = -(double)rand() / RAND_MAX * INIT_WIDTH / 10;
        h_point[i][1] = -(double)rand() / RAND_MAX * INIT_WIDTH / 10;
        h_point[i][2] = -(double)rand() / RAND_MAX * INIT_WIDTH / 10;
      }
  }
  
  checkCudaErrors(cudaMalloc((void**)&d_point, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dv_point, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dst_point, NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&de_point, NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dT_point, NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dJ_point, NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dcolsynctime, NUM_POINTS*NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dcolsyncindex, NUM_POINTS*NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_point, h_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dv_point, v_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dst_point, st_point, NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(de_point, e_point, NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dT_point, T_point, NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dJ_point, J_point, NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcolsynctime, colsynctime, NUM_POINTS*NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcolsyncindex, colsyncindex, NUM_POINTS*NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
}
//CUDA実行関数
void launchGPUKernel(unsigned int num_particles, float (*pos)[3], float (*vec)[3] ,float(*sti),float(*e),float(*T),float(*J), float time, float dt,float (*synctime)[NUM_POINTS],float (*syncindex)[NUM_POINTS])
{
    dim3 grid(Grid_x,Grid_y,Grid_z);
    dim3 block(Block_x,Block_y,Block_z);
    grav_v<<<grid , block>>>(pos, vec, sti, e, T, J, time, dt, synctime, syncindex);
    grav_p<<<grid , block>>>(pos, vec, time, dt);
}
//アニメーション動作
void runGPUKernel(void)
{
  launchGPUKernel(NUM_POINTS, d_point, dv_point ,dst_point, de_point,dT_point,dJ_point, anim_time, anim_dt,dcolsynctime,dcolsyncindex);
  checkCudaErrors(cudaMemcpy(h_point, d_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(v_point, dv_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(st_point, dst_point, NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(e_point, de_point, NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(T_point, dT_point, NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(J_point, dJ_point, NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(colsynctime,dcolsynctime, NUM_POINTS*NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(colsyncindex,dcolsyncindex, NUM_POINTS*NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));

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

void display(void)
{
    double nrml_vec[3];

    light_pos[0] = (float)eye[X];
    light_pos[1] = (float)eye[Y];
    light_pos[2] = (float)eye[Z];
    light_pos[3] = 0.0f;
    //CUDA開始
    runGPUKernel();

    // 光源の設定
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    //glLightfv(GL_LIGHT0, GL_DIFFUSE, light_pos);
    glEnable(GL_LIGHTING);
    glMatrixMode(GL_PROJECTION);
    glFrustum(-1000000, 1000000, -1000000, 1000000, -1000000, 1000000); 
    glLoadIdentity();
    glOrtho(-vision, vision, -vision, vision, -1000, 1000);
    glViewport(0, 0, window_width, window_height);
    defineViewMatrix(phi, theta);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBegin(GL_QUADS);
    
    //球体をポリゴンで作成 TBD メタボール
    for (int k = 0 ; k < NUM_POINTS ; k++)
      {
	for (int i = 0 ; i < dev + 1 ; i ++)
	  {
	    for (int j = 0 ; j < 2 * dev + 1 ; j++)
	      {
		normal(point[i * (dev-1) + j],point[(i + 1) * (dev-1) + j + 1],point[(i+1) * (dev-1) + j],nrml_vec);
		glNormal3dv(nrml_vec);
		glVertex3d(point[i * (dev-1) + j][X] + h_point[k][X], point[i * (dev-1) + j][Y] + h_point[k][Y], point[i * (dev-1) + j][Z] + h_point[k][Z]);
		glVertex3d(point[(i + 1) * (dev-1) + j][X] + h_point[k][X],point[(i + 1) * (dev-1) + j][Y] + h_point[k][Y],point[(i + 1) * (dev-1) + j][Z] + h_point[k][Z]);
		glVertex3d(point[(i + 1) * (dev-1) + j + 1][X] + h_point[k][X], point[(i + 1) * (dev-1) + j + 1][Y] + h_point[k][Y], point[(i + 1) * (dev-1) + j + 1][Z] + h_point[k][Z]);
		glVertex3d(point[i * (dev-1) + j + 1][X] + h_point[k][X],point[i * (dev-1) + j + 1][Y] + h_point[k][Y],point[i * (dev-1) + j + 1][Z] + h_point[k][Z]);
	      }
	  }
      }
    glEnd();
    glutSwapBuffers();
    glutPostRedisplay();
}

void mouse_button(int button, int state, int x, int y)
{
  if ((state == GLUT_DOWN) && (button == GLUT_LEFT_BUTTON))
    motion_p = true;
  else if (state == GLUT_UP)
    motion_p = false;
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
  mouse_old_x = x;
  mouse_old_y = y;
  glutPostRedisplay();
}
void resize (int width, int height)
{
  window_width = width;
  window_height = height;
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
  double yangle,zangle;
  double r;

  point = (double **)malloc(sizeof(double *) * num_points);
  for (int i = 0 ; i < num_points ; i++)
    {
      point[i] = (double *)malloc(sizeof(double) * 3);
    } 
  for (int i = 0 ; i < dev + 1; i ++)
    {
      zangle = i * PI / dev;

      r=R * sin(zangle);
      for (int j = 0 ; j < dev + 1; j++)
	{
	  yangle=j * PI * 2 / dev;

	  point[i * dev + j][X] = r * sin(yangle);
	  point[i * dev + j][Y] = r * cos(yangle);
	  point[i * dev + j][Z] = R * cos(zangle);
	}
    }
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(INIT_WIDTH, INIT_HEIGHT);
    glutCreateWindow("3D CUDA Simulation");
    glutDisplayFunc(display);
    glutReshapeFunc(resize);
    glutMouseFunc(mouse_button);
    glutMotionFunc(mouse_motion);

    setInitialPosition();

    if (!initGL())
      return 1;
 
    glutMainLoop();
    cudaFree(d_point);
    cudaFree(dv_point);
    cudaFree(dst_point);
    cudaFree(de_point);
    cudaFree(dT_point);
    cudaFree(dJ_point);

    cudaDeviceReset();
    for (int i = 0 ; i < num_points ; i++)
      {
	free (point[i]);
	cudaFree(dcolsynctime[i]);
	cudaFree(dcolsyncindex[i]);
      }
    free (point);
    cudaFree(dcolsynctime);
    cudaFree(dcolsyncindex);

    return 0;
}
