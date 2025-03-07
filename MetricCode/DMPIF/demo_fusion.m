%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% This is the demo code for 
%%%% "A New Variational Approach Based on Proximal Deep Injection and
%%%% Gradient Intensity Similarity for Spatio-spectral Image Fusion"
%%%% by Z.-C. Wu, T.-Z. Huang, L.-J. Deng, G. Vivone, J.-Q. Miao, J.-F. Hu, X.-L. Zhao, JSTARS 2020.

%% Clean up workspace
clear;clc
close all;
%% load data which type is double precision and the range is [0 1]
% addpath(genpath('D:\MATLAB_work\DMPIF_peapr'))
addpath(genpath(pwd))
load 'WV3_Rio.mat'
Ori_HRMS = gt;
I_LRMS   = lrms;
I_PAN    = pan;
X_net    = pannet;
clear gt lrms pan pannet;
opts.sensor = 'WV3';          % 'WV3'and 'WV2'etc.
%%  Initialization
opts.lambda = 0.011;
opts.alpha  = 0.50;
opts.eta    = 0.1;
opts.sf     = 4;
sz = size(I_PAN);
[~,~,L]    = size(I_LRMS);
opts.Nways = [sz,L];
opts.sz    = sz;
opts.tol   = 2*1e-4;   
opts.maxit = 22;
opts.Fit   = 10;
% opts.block = 64;
%% return CNN prior to the fusing framework
disp('---------------------------------------Begin the Fusion algorithm---------------------------------------')
t0 = tic; 
X_fin = DMPIF_fusion(I_LRMS, I_PAN, X_net, opts);
time = toc(t0);
fprintf('Time the algorithm is running:  %.2f seconds \n', time)
disp('-------------------------------------End of the Fusion algorithm run-------------------------------------')
%% plotting
close all
location = [65 85 5 25];

figure, showRGB4(Ori_HRMS, Ori_HRMS, location);title('Orginal RGB');

figure, showRGB4(Ori_HRMS, X_fin, location);title('Fusion by DMPIF (RGB)');