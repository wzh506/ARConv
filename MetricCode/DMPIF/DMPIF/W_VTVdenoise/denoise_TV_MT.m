function [X_den,P1,P2]=denoise_TV_MT(Xobs,P1_init, P2_init,opts)
%This function implements the FISTA method for JTV denoising problems. 
%
% INPUT
% Xobs ..............................observed noisy images. [H, W, T]
% OUTPUT
% X_den ........................... The solution of the problem 
%                                            min{||X-Xobs||^2+2*lambda*TV(X)}
% Assigning parameres according to pars and/or default values

% if (flag&&isfield(pars,'epsilon'))
%     epsilon=pars.epsilon;
% else
%     epsilon=1e-4;
% end
% if(flag&&isfield(pars,'print'))
%     prnt=pars.print;
% else
%     prnt=1;
% end
%%
Fit    = opts.Fit;
lambda = opts.lambda;
eta    = opts.eta;
[m,n,T]=size(Xobs);

if(isempty(P1_init))
    for t=1:T
        P1{t}=zeros(m-1,n);    P2{t}=zeros(m,n-1);
        R1{t}=zeros(m-1,n);    R2{t}=zeros(m,n-1);
    end
else
    for t=1:T
        P1{t}=P1_init{t};    P2{t}=P2_init{t};
        R1{t}=P1_init{t};    R2{t}=P2_init{t};
    end
end


tkp1=1;i=0;
D=zeros(m,n,T);
while (i<Fit)
    i=i+1;    
    Pold1=P1;Pold2=P2;    
    tk=tkp1;
    for t=1:T
        temp=Xobs(:,:,t)-(lambda/eta)*Lforward(R1{t}, R2{t}, m, n);
        D(:,:,t)=temp ;        
    end
    % Taking a step towards minus of the gradient
    step0=eta/(8*lambda);
    for t=1:T
        [tQ1, tQ2]=Ltrans(D(:,:,t), m, n);
        P1{t}=R1{t}+step0*tQ1;
        P2{t}=R2{t}+step0*tQ2;
    end
%% ¸üÐÂR S
            A=[P1{1}.^2;zeros(1,n)]+[P2{1}.^2,zeros(m,1)];
            for t=2:T
                A=A+[P1{t}.^2;zeros(1,n)]+[P2{t}.^2,zeros(m,1)];
            end
            A=sqrt(max(A,1));
            for t=1:T
                P1{t}=P1{t}./A(1:m-1,:); 
                P2{t}=P2{t}./A(:,1:n-1);
            end
    %% UpdatingUV and t
    tkp1=(1+sqrt(1+4*tk^2))/2;
    
    step=(tk-1)/tkp1;
    for t=1:T
        R1{t}=P1{t}+step*(P1{t}-Pold1{t});
        R2{t}=P2{t}+step*(P2{t}-Pold2{t});
    end
    
end
X_den=D;
end