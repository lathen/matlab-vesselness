/* Parallel computation of eigenvalues in 3D. This follows appendix G in
 * Gunnar Farnebäck's PhD thesis "Polynomial Expansion for Orientation and
 * Motion Estimation"
 *
 * Elements are passed as arguments t1-t6 according to the layout:
 *     | t1 t2 t3 |
 * T = | t2 t4 t5 | 
 *     | t3 t5 t6 |
 *
 * Author: Gunnar Farnebäck
 *	  Computer Vision Laboratory
 *	  Linköping University, Sweden
 *	  gf@isy.liu.se
 */

#include <cuda.h>

#define PI 3.14159265358979323846

// Eigenvalue device kernel
__global__ void eigenvalues3DKernel(
	const int num_elements,
	const float * T1,
	const float * T2,
	const float * T3,
	const float * T4,
	const float * T5,
	const float * T6,
	float * eig1,
	float * eig2,
	float * eig3
){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_elements) return;

	float t1 = T1[i];
	float t2 = T2[i];
	float t3 = T3[i];
	float t4 = T4[i];
	float t5 = T5[i];
	float t6 = T6[i];

	float tr = (t1 + t4 + t6) / 3.0;

	float a = t1 - tr;
	float b = t4 - tr;
	float c = t6 - tr;

	float d = t2;
	float e = t3;
	float f = t5;

	float p = a*b + a*c + b*c - d*d - e*e - f*f;
	float q = a*f*f + b*e*e + c*d*d - 2.f*d*e*f - a*b*c;
	if (p>0.0) p = 0;
	float beta = sqrtf(-4.0/3.0*p);
	float phi = p*beta;
	if (phi == 0.0) phi = 1.0;
	float gamma = 3.0*q/phi;
	if (gamma > 1.0) gamma = 1.0;
	if (gamma < -1.0) gamma = -1.0;
	float alpha = acosf(gamma)/3.0;

	eig1[i] = tr + beta*__cosf(alpha);
	eig2[i] = tr + beta*__cosf(alpha-2.0*PI/3.0);
	eig3[i] = tr + beta*__cosf(alpha+2.0*PI/3.0);
}
