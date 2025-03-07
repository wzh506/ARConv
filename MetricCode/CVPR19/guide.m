function [Y,P]= guide(M,P, divK, lambda, iter)
%%%% min 1/2*|RX-M|_2^2 + lamda/2*|grad(X)-X_g|_2^2
% Intput----
% M : multispectral image
% P : Pan image
% divK: the resolution difference between the Pan and MS
% lambda: regularization parameter
% iter: the number of iterations one does to minimize the energy

% Output----
% Y: the fused image

%% parameters settings

T = size(M,3);
[m,n] = size(P);
L=1;
x = imresize(M,[m,n]);
y=x;
tnew=1;
Fused = y;

fx=[1,-1];
fy=[1;-1];

par.v1 = 1;
par.v2 =lambda ;
par.v3 = 0.1;
par.r = 2;
par.eps = 1e-8;
par.L = 1;
par.w = 0.25;

%%  FFT Transform
PP = padarray(P,[1,1]*par.r ,'replicate','both');
yp0 = gradient(PP,1);   
yp90 = gradient(PP,3);  
[mm,nn,~]=size(PP);
par.otfFx = psf2otf(fx,[mm,nn]);
par.otfFy = psf2otf(fy,[mm,nn]); 
par.eigsDtD2 = abs(par.otfFx).^2 + abs(par.otfFy ).^2;

%% begin iteration
for k=1:iter
    
    told=tnew;
    xp=x;
      
    for i=1:size(M,3)
      dd(:,:,i) = imresize(x(:,:,i),[m/divK,n/divK]) - M(:,:,i);
      df2(:,:,i) = imresize(dd(:,:,i),[m,n],'bicubic');
    end

    yg = y -  df2/L;            
    x = padarray(x,[1,1]*par.r,'replicate','both');  
    
    for i = 1:T       
        x(:,:,i) = argminx(x(:,:,i),padarray(yg(:,:,i) ,[1,1]*par.r,'replicate','both'),yp0,yp90,par);
    end
        
     x = x(par.r+1:end-par.r,par.r+1:end-par.r,:);     
     tnew=(1+sqrt(1+4*told^2))/2;
     y=x+((told-1)/tnew)*(x-xp);
 
end

Y = x;

