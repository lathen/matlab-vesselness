function V = vesselness3D_Sato_CUDA(data, k)

V = parallel.gpu.GPUArray.zeros(size(data), 'single');
eig1 = parallel.gpu.GPUArray.zeros(size(data), 'single');
eig2 = parallel.gpu.GPUArray.zeros(size(data), 'single');
eig3 = parallel.gpu.GPUArray.zeros(size(data), 'single');

g = gpuDevice;

for ind = 1:length(k.scale)
    fxx = conv3_CUDA_separable_kernel(data, k.scale(ind).G2, k.scale(ind).G0, k.scale(ind).G0, k.scale(ind).conv);
    fxy = conv3_CUDA_separable_kernel(data, k.scale(ind).G1, k.scale(ind).G1, k.scale(ind).G0, k.scale(ind).conv);
    fxz = conv3_CUDA_separable_kernel(data, k.scale(ind).G1, k.scale(ind).G0, k.scale(ind).G1, k.scale(ind).conv);
    fyy = conv3_CUDA_separable_kernel(data, k.scale(ind).G0, k.scale(ind).G2, k.scale(ind).G0, k.scale(ind).conv);
    fyz = conv3_CUDA_separable_kernel(data, k.scale(ind).G0, k.scale(ind).G1, k.scale(ind).G1, k.scale(ind).conv);
    fzz = conv3_CUDA_separable_kernel(data, k.scale(ind).G0, k.scale(ind).G0, k.scale(ind).G2, k.scale(ind).conv);

    [eig1, eig2, eig3] = eval_kernel(k.eig, fxx, fxy, fxz, fyy, fyz, fzz, ...
                                          eig1, eig2, eig3);
    
    %g.FreeMemory/(1024*1024)
    
    clear fxx;
    clear fxy;
    clear fxz;
    clear fyy;
    clear fyz;
    clear fzz;
    
    gamma = 1;
    alpha = 0.25;
    
    v = parallel.gpu.GPUArray.zeros(size(data), 'single');
    v = eval_kernel(k.vesselness, eig1, eig2, eig3, v, gamma, alpha);
    
    V = eval_kernel(k.max, V, v);
    clear v;
    
end
