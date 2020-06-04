#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define X 0
#define Y 1
#define Z 2
#define nump 20
#define NUM_POINTS (nump * nump * nump)
#define PI 3.141592653589793
#define R 0.2
#define INIT_WIDTH 750
#define INIT_HEIGHT 750
#define GRAV 500
#define dist 2
#define vision 3
#define Grid_x nump
#define Grid_y nump
#define Grid_z 1
#define Block_x nump
#define Block_y 1
#define Block_z 1

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
float anim_dt = 0.00000001f;
double phi = 30.0;
double theta = 30.0;
float light_pos[4];
int mouse_old_x, mouse_old_y;
bool motion_p;
double eye[3];
double center[3] = {0.0, 0.0, 0.0};
double up[3];

float (*d_point)[3];
float (*dv_point)[3];

GLuint vbo;

__global__ void grav_v(float (*pos)[3], float(*vec)[3] , float time, float dt);
__global__ void grav_p(float (*pos)[3], float(*vec)[3] , float time, float dt);

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

__global__ void grav_v(float (*pos)[3], float(*vec)[3] , float time, float dt)
{
  double xn,yn,zn,vx,vy,vz,dis,sq;

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

  for (int i = 0 ; i < NUM_POINTS; i++)
    {
      sq = pow((double)(xn-pos[i][0]),2) + pow((double)(yn-pos[i][1]),2) + pow((double)(zn-pos[i][2]),2);
      dis = sqrt(sq);
      if (dis > dist * R)
	{
	  vx = vx + (pos[i][0]-xn)/dis/sq*R*R*R*GRAV;
	  vy = vy + (pos[i][1]-yn)/dis/sq*R*R*R*GRAV;
	  vz = vz + (pos[i][2]-zn)/dis/sq*R*R*R*GRAV;
	}
    }
  vec[index][0] = vx;
  vec[index][1] = vy;
  vec[index][2] = vz;
}

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

void launchGPUKernel(unsigned int num_particles, float (*pos)[3], float (*vec)[3] , float time, float dt)
{
    dim3 grid(Grid_x,Grid_y,Grid_z);
    dim3 block(Block_x,Block_y,Block_z);

    grav_v<<<grid , block>>>(pos, vec, time, dt);
    grav_p<<<grid , block>>>(pos, vec, time, dt);

}

// 粒子を初期位置に配置．
void setInitialPosition(void)
{
  unsigned int i, j, k ;
  unsigned int count = 0;

	for (i = 0; i < nump; i++) {
	  for (j = 0; j < nump; j++) {
	    for (k = 0; k < nump; k++) {
	      h_point[count][0] = -0.5 + (double)i / nump ;
	      h_point[count][1] = -0.5 + (double)j / nump ;
	      h_point[count][2] = -0.5 + (double)k / nump ;
	      count++;
	    }
	  }
	}
	for (i = 0; i < NUM_POINTS; i++) {
	  v_point[i][0] = 0;
	  v_point[i][1] = 0;
	  v_point[i][2] = 0;
	}

  /*デバイスメモリ領域の確保*/
  checkCudaErrors(cudaMalloc((void**)&d_point, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&dv_point, 3 * NUM_POINTS * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_point, h_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dv_point, v_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));

}

void runGPUKernel(void)
{
  launchGPUKernel(NUM_POINTS, d_point, dv_point , anim_time, anim_dt);
  checkCudaErrors(cudaMemcpy(h_point, d_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(v_point, dv_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, NUM_POINTS * 3 * sizeof(float), h_point, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  anim_time += anim_dt;
}

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

    light_pos[0] = (float)eye[X];
    light_pos[1] = (float)eye[Y];
    light_pos[2] = (float)eye[Z];
    light_pos[3] = 1.0f;

    runGPUKernel();

    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glEnable(GL_LIGHTING);

    glMatrixMode(GL_PROJECTION);
    glFrustum(-1000000, 1000000, -1000000, 1000000, -1000000, 1000000); 

    glLoadIdentity();

    glOrtho(-vision, vision, -vision, vision, -1000, 1000);
    glViewport(0, 0, window_width, window_height);
    defineViewMatrix(phi, theta);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, 0);    
    glDrawArrays(GL_POINTS , 0, 0, NUM_POINTS);
    glDisableClientState(GL_VERTEX_ARRAY); 
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutSwapBuffers();
    glutPostRedisplay();
}

void resize (int width, int height)
{
  window_width = width;
  window_height = height;
}

bool initGL(void)
{
  glClearColor(1.0f, 1.0f , 1.0f, 1.0f);
  glEnable(GL_DEPTH_TEST);
  glClearDepth(1.0);
  glDepthFunc(GL_LESS);
  glEnable(GL_LIGHT0);

  glewInit();
  createVBO(&vbo, NUM_POINTS * 3 * sizeof(float) );

  return true;
}

int main(int argc, char** argv)
{

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

    glutInitWindowSize(INIT_WIDTH, INIT_HEIGHT);
    glutCreateWindow("3D CUDA Simulation");
    glutDisplayFunc(display);
    glutReshapeFunc(resize);

    setInitialPosition();

    if (!initGL())
      return 1;
 
    glutMainLoop();
    cudaFree(d_point);
    cudaFree(dv_point);

    deleteVBO(&vbo);

    cudaDeviceReset();
    return 0;
}
