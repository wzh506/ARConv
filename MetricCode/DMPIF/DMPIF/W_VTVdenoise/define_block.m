function de_block = define_block(Y_b, P_b)
[R,C,S]   = size(Y_b);
de_block = zeros([R,C,S]);
 for i=1:S
    temp=Y_b(:,:,i);
    Av_light=mean(temp(:));
    multiplier=Av_light/mean(P_b(:));
    de_block(:,:,i)=multiplier*P_b;
 end
end