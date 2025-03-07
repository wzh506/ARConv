function [X_out]  =  DMPIF_fusion(LRMS, PAN, Xhf, opts)
%% Initiation
maxit = opts.maxit;
tol   = opts.tol;              
Nways = opts.Nways;
eta   = opts.eta ;
sf    = opts.sf;
% block = opts.block;0.1
Thet  = zeros(Nways); 
X_init = interp23tap(LRMS, sf);
W_init = zeros(Nways);
X = X_init;
W = W_init;
%P_3D=define_P(LMS ,PAN ,block);
%P_3D=repmat(PAN,[1 1 8]);
P_3D = define_Pnew(LRMS ,PAN);
par  = FFT_kernel(opts);
%%
X_k = X;
for it = 1:maxit
    % update X
    X = X_subproblem(LRMS, Xhf, P_3D, W, Thet, par, opts);
    % update W
    temp = P_3D+Thet/eta-X;
      if (it==1)
          [W_3D, P1, P2] = denoise_TV_MT(temp, [], [], opts);
      else
          [W_3D, P1, P2] = denoise_TV_MT(temp, P1, P2, opts);
      end
     W = W_3D;
    % update thet
    Thet = Thet+eta*(P_3D-X-W);
    %% stopping criterion
    Rel_Err = norm(Unfold(X-X_k,Nways,3) ,'fro')/norm(Unfold(X_k,Nways,3),'fro');
    X_k = X;
    if Rel_Err < tol  
        break;
    end   
end
 X_out=X;
end
   