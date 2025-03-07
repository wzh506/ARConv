function X=Lforward(P1, P2, m, n)

      X=zeros(m,n);
      X(1:m-1,:)=P1;
      X(:,1:n-1)=X(:,1:n-1)+P2;
      X(2:m,:)=X(2:m,:)-P1;
      X(:,2:n)=X(:,2:n)-P2;
 end
 