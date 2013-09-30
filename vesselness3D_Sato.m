function [V, S, B] = vesselness3D_Sato(data,scales)

V = zeros(size(data));
if nargout > 1
    S = zeros(size(data));
end
if nargout > 2
    B = zeros(size(data));
end

for scale = scales
    sigma = sqrt(2)^(scale-1);
    n = round(8*sigma);
    if mod(n,2) == 0
        n = n + 1;
    end
    
    G0 =           gaussiankernel(n,sigma,0);
    G1 = sigma   * gaussiankernel(n,sigma,1);
    G2 = sigma^2 * gaussiankernel(n,sigma,2);
    
    %figure;
    %subplot(1,3,1), plot(G0);
    %subplot(1,3,2), plot(G1);
    %subplot(1,3,3), plot(G2);
    
    disp(['Filtering scale ',int2str(scale),' with sigma ',num2str(sigma),'...']);
    
    % fxx
    filters{1}{1} = G2;
    filters{1}{2} = G0';
    filters{1}{3} = reshape(G0,[1 1 length(G0)]);
    
    % fxy
    filters{2}{1} = G1;
    filters{2}{2} = G1';
    filters{2}{3} = reshape(G0,[1 1 length(G0)]);
    
    % fxz
    filters{3}{1} = G1;
    filters{3}{2} = reshape(G1,[1 1 length(G1)]);
    filters{3}{3} = G0';
    
    % fyy
    filters{4}{1} = G2';
    filters{4}{2} = G0;
    filters{4}{3} = reshape(G0,[1 1 length(G0)]);
    
    % fyz
    filters{5}{1} = G1';
    filters{5}{2} = reshape(G1,[1 1 length(G1)]);
    filters{5}{3} = G0;
    
    % fzz
    filters{6}{1} = reshape(G2,[1 1 length(G2)]);
    filters{6}{2} = G0;
    filters{6}{3} = G0';
    
    f = cell(6,1);
    parfor comp = 1:6
        f{comp} = data;
        for filter = 1:3
            f{comp} = imfilter(f{comp}, filters{comp}{filter}, ...
                                'replicate', 'same', 'conv');
        end
    end
    
    [eig1, eig2, eig3] = eigenvalues3D(f{1}, f{2}, f{3}, f{4}, f{5}, f{6});
    
    gamma = 1;
    alpha = 0.25;
    
    v = abs(eig3).*phi(eig2,eig3,gamma).*omega(eig1,eig2,gamma,alpha);
    v(eig2 >= 0) = 0;
    
    V = max(V,v);
    
    if nargout > 1
        s = abs(eig3).*omega(eig2,eig3,gamma,alpha).*omega(eig1,eig3,gamma,alpha);
        s(eig3 >= 0) = 0;
        
        S = max(S,s);
    end
    
    if nargout > 2
        b = abs(eig3).*phi(eig2,eig3,gamma).*phi(eig1,eig2,gamma);
        b(eig1 >= 0) = 0;
        
        B = max(B,b);
    end
end

end

function p = phi(eig1, eig2, gamma)
    p = (eig1./eig2).^gamma;
    p(eig2 >= 0) = 0;
end

function o = omega(eig1, eig2, gamma, alpha)
    o = zeros(size(eig1));
    
    tmp = (1 + eig1./abs(eig2)).^gamma;
    o(eig1 <= 0) = tmp(eig1 <= 0);
    
    tmp = (1 - alpha*eig1./abs(eig2)).^gamma;
    o(eig1 > 0 & eig1 < abs(eig2)/alpha) = tmp(eig1 > 0 & eig1 < abs(eig2)/alpha);   
end
