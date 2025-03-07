function X_out = X_subproblem(LRMS, Xhf, P_3D, W, Thet, par, opts)
sz    = opts.sz;
alpha = opts.alpha;
Nways = opts.Nways;
eta   = opts.eta ;
sf    = opts.sf;

H1 = (alpha+eta/2)*eye(Nways(3));

H3_highdims = par.B(par.ST(LRMS))+alpha*Xhf+(eta/2)*(P_3D+W)-Thet/2;
H3 = Unfold(H3_highdims, Nways, 3);
 
X_mode = Sylvester(H1, par.fft_B, par.fft_BT, sf , sz(1)/sf, sz(2)/sf, H3); %% Sylvester equation (5)
X_out = Fold(X_mode, Nways, 3); 

X_out(X_out>1)=1; 
X_out(X_out<0)=0;
end 