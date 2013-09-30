
%% Load data
load spiral;
data = single(data);

%% Step to the parent directory and set paths
S = pwd;
cd ..
P = path;
addpath('CUDA-conv3');
addpath('CUDA-conv3/utils');
addpath('lib');


%% Determine kernel radius for largest kernel (given scale)
%  and pad dataset to remove boundary effects and match the size to be a
%  multiple of 32 on GPU
scale = 4;
sigma = sqrt(2)^(scale-1);
n = round(8*sigma);
if mod(n,2) == 0
    n = n + 1;
end
kernel_radius = (n-1)/2;

[dataExt, offset] = extendvolume(data, 32, kernel_radius);
dataGPU = gpuArray(dataExt);


%% Compute vesselness on CPU
% Start workers for parallel processing
if matlabpool('size') == 0
    matlabpool open;
end
t = tic;
V_CPU = vesselness3D_Sato(data,1:scale);
toc(t)
matlabpool close;


%% Compute vesselnes on GPU
k = vesselness3D_Sato_CUDA_make(1:scale);
tic;
V_kernelGPU = vesselness3D_Sato_CUDA(dataGPU, k);
toc;
V_kernel = gather(V_kernelGPU);
V_kernel = V_kernel(offset(1):offset(1)+size(data,1)-1, ...
                    offset(2):offset(2)+size(data,2)-1, ...
                    offset(3):offset(3)+size(data,3)-1);

%%
min(V_kernel(:)), max(V_kernel(:))
min(V_CPU(:)), max(V_CPU(:))

max(max(max(abs(V_kernel-V_CPU))))

%%
viewer3d(V_kernel,V_CPU);

%% Step back...
cd(S);

% Reset the path
path(P);
