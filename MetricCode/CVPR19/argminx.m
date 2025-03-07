function x = argminx(x,y,yp0,yp90,par)

x0 = gradient(x,1);    
x90 = gradient(x,3);  

xg0 = guidedfilter(yp0, x0, par.r, par.eps);  
xg90 = guidedfilter(yp90, x90, par.r, par.eps);
        
weights = par.L * par.v1 + par.v2 * par.eigsDtD2;

fftdatay = fft2(y);
fftdatax0 = conj(par.otfFx) .*  fft2(xg0);  
fftdatax90 = conj(par.otfFy) .* fft2(xg90); 

fftdata2 = fftdatax0 + fftdatax90;  

S = ( fftdatay * par.v1 * par.L +  par.v2 * fftdata2 )./weights;

x = abs(ifft2(S));

