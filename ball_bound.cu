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

#define PI 3.141592653589793

#define cap 1000
#define ref 0.6
#define temp 4273
#define visc 1000000000
#define GRAV (6.674*0.00000000000000000001)
#define density (2.5 * 1000000000000)
#define M (4 / 3 * PI * reluc*reluc*reluc* density)

#define reluc 15
#define dev 12
#define nump 15
#define NUM_POINTS (nump * nump * nump)

#define X 0
#define Y 1
#define Z 2
#define INIT_WIDTH 500
#define INIT_HEIGHT 500
#define vision 20
#define Grid_x nump
#define Grid_y nump
#define Grid_z 1
#define Block_x nump
#define Block_y 1
#define Block_z 1

unsigned int num_points = (dev + 1) * (dev + 1);
unsigned int window_width = 150;
unsigned int window_height = 150;
double init_left = -10000;
double init_right = 10000;
double init_bottom = -10000;
double init_top = 10000;
double left, right, bottom, top;
float h_point[NUM_POINTS][3];
float v_point[NUM_POINTS][3];
float anim_time = 0.0f;
float anim_dt = 0.000000000001;
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

__global__ void grav_v(float (*pos)[3], float(*vec)[3] , float time, float dt);
__global__ void grav_p(float (*pos)[3], float(*vec)[3] , float time, float dt);
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
int comp (const void *a, const void *b){
  return *(int *)a-*(int *)b;
}

//重力影響後の速度を決定
__global__ void grav_v(float (*pos)[3],float(*vec)[3], float time, float dt)
{
  double xn,yn,zn,vx,vy,vz,dis,sq;
  unsigned int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned int index = ( blockDim.x * (Grid_x - 1) + blockDim.x ) * ( blockDim.y * (Grid_y - 1) + blockDim.y ) * thread_idz + ( blockDim.x * (Grid_x - 1) + blockDim.x ) * thread_idy + thread_idx ;
  float v_buff[NUM_POINTS][3]={0};

  float alive[NUM_POINTS]={1};
  float r[NUM_POINTS]={reluc};
  float m[NUM_POINTS]={M};
  float s[NUM_POINTS]={visc};
  float e[NUM_POINTS]={ref};
  float T[NUM_POINTS]={temp};
  float J[NUM_POINTS]={cap*M*temp};

  float coltime[NUM_POINTS][2]={0};
  int colflag=0;
  xn = pos[index][0];
  yn = pos[index][1];
  zn = pos[index][2];
  vx = vec[index][0];
  vy = vec[index][1];
  vz = vec[index][2];
  v_buff[index][0]=vx;
  v_buff[index][1]=vy;
  v_buff[index][2]=vz;

  for (int i = 0 ; i < NUM_POINTS; i++)
    {
      sq = pow((double)(xn-pos[i][0]),2) + pow((double)(yn-pos[i][1]),2) + pow((double)(zn-pos[i][2]),2);
      dis = sqrt(sq);
      if (dis > r[index] + r[i])
	{
          //運動エネルギーと内部エネルギーの交換 
	  J[i]-=0.5*m[index]*fabs((pow((double)vx,2)+pow((double)vy,2)+pow((double)vz,2))-(pow((double)(vx + ((pos[i][0]-xn)/dis)*(GRAV*m[index]*m[i]/sq)),2)+pow((double)(vy + ((pos[i][1]-yn)/dis)*(GRAV*m[index]*m[i]/sq)),2)+pow((double)(vz + ((pos[i][2]-zn)/dis)*(GRAV*m[index]*m[i]/sq)),2)));
	  //速度更新
	  vx = vx + ((pos[i][0]-xn)/dis)*(GRAV*m[index]*m[i]/sq);
	  vy = vy + ((pos[i][1]-yn)/dis)*(GRAV*m[index]*m[i]/sq);
	  vz = vz + ((pos[i][2]-zn)/dis)*(GRAV*m[index]*m[i]/sq);
	}
      else 
	{
	  if (i != index)
	    {
	      colflag=1;
	      //衝突するまでの時間を記録　TBD　法線方向に直す
              coltime[i][1]=(float)i;
	      coltime[i][0]=(float)(sq/(pow((double)(vx-vec[index][0]),2)+pow((double)(vy-vec[index][1]),2)+pow((double)(vz-vec[index][2]),2)));
	    }
	}
    }
  if(colflag==1){
    //衝突までの時間をインデックスソート
    qsort(coltime, sizeof(coltime)/2/sizeof(float), 2*sizeof(float),comp);
    for (int i=NUM_POINTS-1 ; i>=0; i--){
      int colindex=(int)coltime[i][1];
      //反発係数は小さいほうを優先
      float repul=e[index];
      if (e[colindex] < e[index]) {
	repul=e[colindex];
      }
      //速度更新
      v_buff[index][0]=((1+repul)*m[colindex]*vec[colindex][0]+(m[index]-repul*m[colindex])*v_buff[index][0])/(m[index]+m[colindex]);
      v_buff[index][1]=((1+repul)*m[colindex]*vec[colindex][1]+(m[index]-repul*m[colindex])*v_buff[index][1])/(m[index]+m[colindex]);
      v_buff[index][2]=((1+repul)*m[colindex]*vec[colindex][2]+(m[index]-repul*m[colindex])*v_buff[index][2])/(m[index]+m[colindex]);
      //衝突エネルギーの熱エネルギー変換
      float Energy=0.5*(1-repul*repul)*(m[index]*())
      //粘性と反発係数の更新

      //反発係数がしきい値以下なら吸収
      
    }
  }
  __syncthreads();
  vec[index][0] = vx + v_buff[index][0];
  vec[index][1] = vy + v_buff[index][1];
  vec[index][2] = vz + v_buff[index][2];
  //内部エネルギーと運動エネルギーから熱エネルギー更新                                                                
  T[index]=(J[index]-0.5*m[index]*(pow((double)vec[index][0],2)+pow((double)vec[index][1],2)+pow((double)vec[index][2],2)))/m[index]/caps;
  
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
	h_point[i][0] = (double)rand() / RAND_MAX * INIT_WIDTH / 20;
        h_point[i][1] = (double)rand() / RAND_MAX * INIT_WIDTH / 20;
        h_point[i][2] = (double)rand() / RAND_MAX * INIT_WIDTH / 20;
      }
    else
      {
	h_point[i][0] = -(double)rand() / RAND_MAX * INIT_WIDTH / 20;
        h_point[i][1] = -(double)rand() / RAND_MAX * INIT_WIDTH / 20;
        h_point[i][2] = -(double)rand() / RAND_MAX * INIT_WIDTH / 20;
      }
  }
  for (int i = 0; i < NUM_POINTS; i++) {
    v_point[i][0] = 0;
    v_point[i][1] = 0;
    v_point[i][2] = 0;
  }
  checkCudaErrors(cudaMalloc((void**)&d_point, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dv_point, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_point, h_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dv_point, v_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
}
//CUDA実行関数
void launchGPUKernel(unsigned int num_particles, float (*pos)[3], float (*vec)[3] , float time, float dt)
{
    dim3 grid(Grid_x,Grid_y,Grid_z);
    dim3 block(Block_x,Block_y,Block_z);
    grav_v<<<grid , block>>>(pos, vec, time, dt);
    grav_p<<<grid , block>>>(pos, vec, time, dt);
}
//アニメーション動作
void runGPUKernel(void)
{
  launchGPUKernel(NUM_POINTS, d_point, dv_point , anim_time, anim_dt);
  checkCudaErrors(cudaMemcpy(h_point, d_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(v_point, dv_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
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
    
    //球体をポリゴンで作成
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
    cudaDeviceReset();
    for (int i = 0 ; i < num_points ; i++)
      {
	free (point[i]);
      }
    free (point);
    return 0;
}
