
#include <cuda.h>

__device__ float phi(float eig1, float eig2, float gamma) {
	if (eig1 < 0.f)
		return __powf(eig1/eig2, gamma);

	return 0.f;
}

__device__ float omega(float eig1, float eig2, float gamma, float alpha) {
	eig2 = abs(eig2);

	if (eig1 <= 0.f)
		return __powf(1.f + eig1/eig2, gamma);

	if (eig1 < eig2/alpha)
		return __powf(1.f - alpha*eig1/eig2, gamma);

	return 0.f;
}


// Vesselness device kernel
__global__ void vesselness3DKernel(
	const int num_elements,
	const float * eig1,
	const float * eig2,
	const float * eig3,
	float * V,
	float gamma,
	float alpha
){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_elements) return;

	if (eig2[i] < 0)
		V[i] = abs(eig3[i]) * phi(eig2[i],eig3[i],gamma) * omega(eig1[i],eig2[i],gamma,alpha);
	else
		V[i] = 0;
}
