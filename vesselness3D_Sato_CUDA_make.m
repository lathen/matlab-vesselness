function k = vesselness3D_Sato_CUDA_make(scales, threads_per_block)

if nargin < 2
    threads_per_block = 1024;
end

script_dir = fileparts(mfilename('fullpath'));

% Save the path and add the utils, kernels and conv3 directory
P = path;
addpath(fullfile(script_dir, 'CUDA-conv3'));
addpath(fullfile(script_dir, 'CUDA-conv3', 'utils'));

% Create kernels for all scales
for ind = 1:length(scales)
    sigma = sqrt(2)^(scales(ind)-1);
    n = round(8*sigma);
    if mod(n,2) == 0
        n = n + 1;
    end
    
    k.scale(ind).G0 = gpuArray(single(gaussiankernel(n,sigma,0)));
    k.scale(ind).G1 = gpuArray(single(sigma * gaussiankernel(n,sigma,1)));
    k.scale(ind).G2 = gpuArray(single(sigma^2 * gaussiankernel(n,sigma,2)));
    
    k.scale(ind).conv = conv3_CUDA_separable_kernel_make((n-1)/2);
end

% Create eigenvalues kernel
k.eig = kernel_make('kernels\eigenvalues3D_kernel', threads_per_block);

% Create max kernel
k.max = kernel_make('kernels\volume_max_kernel', threads_per_block);

% Create vesselness kernel
k.vesselness = kernel_make('kernels\vesselness3D_Sato_kernel', threads_per_block);

% Reset the path
path(P);
