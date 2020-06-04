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

#define X 0
#define Y 1
#define Z 2
#define nump 16
#define NUM_POINTS (nump * nump * nump)
#define DIS_POINTS (NUM_POINTS * NUM_POINTS)
#define PI 3.141592653589793
#define R 0.5
#define dev 10
#define INIT_WIDTH 750
#define INIT_HEIGHT 750
#define GRAV 100000
#define DIST 20
#define vision 50
#define Grid_x nump*nump
#define Grid_y nump*nump
#define Grid_z nump
#define Block_x nump
#define Block_y 1
#define Block_z 1
#define Grid_a nump
#define Grid_b nump
#define Grid_c 1
#define Block_a nump
#define Block_b 1
#define Block_c 1

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
float anim_dt = 0.00000001f;
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
float (*dis_point)[3];

__global__ void grav_v(float (*pos)[3], float(*dist)[3]);
__global__ void grav_s(float (*vec)[3], float(*dist)[3]);
__global__ void grav_p(float (*pos)[3], float(*vec)[3], float dt);

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

__global__ void grav_v(float (*pos)[3], float(*dist)[3])
{
  double dis,sq;
  unsigned  int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned  int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned  int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned  int thread = ( blockDim.x * (Grid_x - 1) + blockDim.x ) * ( blockDim.y * (Grid_y - 1) + blockDim.y ) * thread_idz + ( blockDim.x * (Grid_x - 1) + blockDim.x ) * thread_idy + thread_idx ;
  unsigned  int index = thread / NUM_POINTS;
  unsigned  int index2 = thread % NUM_POINTS;

  sq = pow((double)(pos[index][0]-pos[index2][0]),2) + pow((double)(pos[index][1]-pos[index2][1]),2) + pow((double)(pos[index][2]-pos[index2][2]),2);
  dis = sqrt(sq);
  if (dis > DIST * R)
    {
      dist[thread][0] = (pos[index2][0]-pos[index][0])/dis/sq*R*R*R*GRAV;
      dist[thread][1] = (pos[index2][1]-pos[index][1])/dis/sq*R*R*R*GRAV;
      dist[thread][2] = (pos[index2][2]-pos[index][2])/dis/sq*R*R*R*GRAV;
    }

}

__global__ void grav_s(float (*vec)[3], float(*dist)[3])
{
  unsigned  int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned  int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned  int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned  int index = ( blockDim.x * (Grid_a - 1) + blockDim.x ) * ( blockDim.y * (Grid_b - 1) + blockDim.y ) * thread_idz + ( blockDim.x * (Grid_a - 1) + blockDim.x ) * thread_idy + thread_idx ;

  for (int i = 0; i < NUM_POINTS; i++)
    {
      vec[index][0] += dist[index * NUM_POINTS + i][0];
      vec[index][1] += dist[index * NUM_POINTS + i][1];
      vec[index][2] += dist[index * NUM_POINTS + i][2];
    }
}

__global__ void grav_p(float (*pos)[3], float(*vec)[3] , float dt)
{
  double xn,yn,zn,vx,vy,vz;
  unsigned  int thread_idx = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned  int thread_idy = threadIdx.y+blockDim.y*blockIdx.y;
  unsigned  int thread_idz = threadIdx.z+blockDim.z*blockIdx.z;
  unsigned  int index = ( blockDim.x * (Grid_a - 1) + blockDim.x ) * ( blockDim.y * (Grid_b - 1) + blockDim.y ) * thread_idz + ( blockDim.x * (Grid_a - 1) + blockDim.x ) * thread_idy + thread_idx ;

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

void launchGPUKernel(float (*pos)[3], float (*vec)[3] , float (*dist)[3], float dt)
{
    dim3 grid1(Grid_x,Grid_y,Grid_z);
    dim3 block1(Block_x,Block_y,Block_z);
    dim3 grid2(Grid_a,Grid_b,Grid_c);
    dim3 block2(Block_a,Block_b,Block_c);

    grav_v<<<grid1 , block1>>>(pos, dist);
    grav_s<<<grid2 , block2>>>(vec, dist);
    grav_p<<<grid2 , block2>>>(pos, vec, dt);
}

// 粒子を初期位置に配置．
void setInitialPosition(void)
{
  unsigned  int i, j, k ;
  unsigned  int count = 0;

	for (i = 0; i < nump; i++) {
	  for (j = 0; j < nump; j++) {
	    for (k = 0; k < nump; k++) {
	      h_point[count][0] = -(nump * R) + (double)i * 2 * R ;
	      h_point[count][1] = -(nump * R) + (double)j * 2 * R ;
	      h_point[count][2] = -(nump * R) + (double)k * 2 * R ;
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
 checkCudaErrors(cudaMalloc((void**)&dis_point, 3 * DIS_POINTS * sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_point, h_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dv_point, v_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyHostToDevice));

}

void runGPUKernel(void)
{
  launchGPUKernel(d_point, dv_point, dis_point, anim_dt);
  checkCudaErrors(cudaMemcpy(h_point, d_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(v_point, dv_point, 3 * NUM_POINTS * sizeof(float) , cudaMemcpyDeviceToHost));


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
    double nrml_vec[3];

    light_pos[0] = (float)eye[X];
    light_pos[1] = (float)eye[Y];
    light_pos[2] = (float)eye[Z];
    light_pos[3] = 1.0f;

    runGPUKernel();


    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_pos);
    glEnable(GL_LIGHTING);

    glMatrixMode(GL_PROJECTION);
    glFrustum(-1000000, 1000000, -1000000, 1000000, -1000000, 1000000); 

    glLoadIdentity();

    glOrtho(-vision, vision, -vision, vision, -1000, 1000);
    glViewport(0, 0, window_width, window_height);
    defineViewMatrix(phi, theta);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBegin(GL_QUADS);
    for (int i = 0 ; i < dev + 1 ; i ++)
      {
	for (int j = 0 ; j < 2 * dev + 1 ; j++)
	  {
	    normal(point[i * (dev-1) + j],point[(i + 1) * (dev-1) + j + 1],point[(i+1) * (dev-1) + j],nrml_vec);
	    glNormal3dv(nrml_vec);
	    for (int k = 0 ; k < NUM_POINTS ; k++)
	      {
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

    setInitialPosition();

    if (!initGL())
      return 1;
 
    glutMainLoop();
    cudaFree(d_point);
    cudaFree(dv_point);
    cudaFree(dis_point);
    cudaDeviceReset();
    for (int i = 0 ; i < num_points ; i++)
      {
	free (point[i]);
      }
    free (point);
    return 0;
}
