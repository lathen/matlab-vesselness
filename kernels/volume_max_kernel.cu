
#include <cuda.h>

__global__ void volume_max(
	const int num_elements,
	float * v1,
	const float * v2
){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_elements) return;

	if (v2[i] > v1[i])
		v1[i] = v2[i];
}
