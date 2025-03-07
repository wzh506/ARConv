function de_P = define_P(Y ,P, B_size)
[nr, nc, ns]= size(Y);
if (mod(nr,B_size)==0)&&(mod(nc,B_size)==0)
    gridx = 1:B_size:nc-B_size+1;
    gridy = 1:B_size:nr-B_size+1;
    de_P  = zeros([nr, nc, ns]);
    for i=1:length(gridx)
        for j=1:length(gridy)
            de_P(i:i+B_size-1,j:j+B_size-1,:)= define_block(Y(i:i+B_size-1,j:j+B_size-1,:),P(i:i+B_size-1,j:j+B_size-1));
        end
    end
%     de_P(de_P>1)=1;
%     de_P(de_P<0)=0;
else
    error('The size of the PAN image should be an integer multiple of the size of the block!')
end
end


