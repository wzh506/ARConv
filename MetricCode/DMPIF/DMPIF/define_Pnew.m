function de_block = define_Pnew(Y_b, P_b)
[~,~,S]   = size(Y_b);
%de_block = zeros([R,C,S]);
 for i=1:S
    temp=Y_b(:,:,i);
    Av_light=mean(temp(:));
    multiplier=Av_light/mean(P_b(:));
    de_block(:,:,i)=multiplier*P_b;
 end
%      temp=de_block;
%      temp=MTF_im(temp,'WV3','',4);
%      de_block(:,:,[1 2])=temp(:,:,[1 2]);
end