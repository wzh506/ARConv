function [psnr, ssim] = Quality_ass(imagery1, imagery2)
%==========================================================================
% Evaluates the quality assessment indices for two tensors.
%
% Syntax:
%   [psnr, ssim] = Quality_ass(imagery1, imagery2)
%
% Input:
%   imagery1 - the reference tensor
%   imagery2 - the target tensor

% NOTE: the tensor is a M*N*K array and DYNAMIC RANGE [0, 255]. 

% Output:
%   psnr - Peak Signal-to-Noise Ratio
%   ssim - Structure SIMilarity

% See also StructureSIM, FeatureSIM
%==========================================================================
Nway = size(imagery1);
psnr = zeros(Nway(3),1);
ssim = zeros(Nway(3),1);
for i = 1:Nway(3)
    psnr(i) = psnr_index(imagery1(:, :, i), imagery2(:, :, i));
    ssim(i) = ssim_index(imagery1(:, :, i), imagery2(:, :, i));
end
psnr = mean(psnr);
ssim = mean(ssim);

