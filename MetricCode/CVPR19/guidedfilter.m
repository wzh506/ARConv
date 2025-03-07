function   q= guidedfilter(I, p, r, eps)
%   - guidance image: I (should be a single channel image)
%   - filtering input image: p (should be a single channel image)
%   - local window radius: r
%   - regularization parameter: eps

thr1 = 1e-5;  %  threshold setting
[hei, wid] = size(I);
N = boxfilter(ones(hei, wid), r); 
mean_I = boxfilter(I, r) ./ N;
mean_p = boxfilter(p, r) ./ N;
mean_Ip = boxfilter(I.*p, r) ./ N;
cov_Ip = mean_Ip - mean_I .* mean_p; 
mean_II = boxfilter(I.*I, r) ./ N;
var_I = mean_II - mean_I .* mean_I;
index = var_I > thr1;
a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;  
a = a.*index;
b = mean_p - a .* mean_I; % Eqn. (6) in the paper;

mean_a = boxfilter(a, r) ./ N;  % Eqn. (18) in the paper;  
mean_b = boxfilter(b, r) ./ N;  % Eqn. (19) in the paper;  

q = mean_a .* I + mean_b; % Eqn. (20) in the paper;

end