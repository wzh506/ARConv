   function [P1, P2]=Ltrans(X, m, n)

      
          P1=X(1:m-1,:)-X(2:m,:);
          P2=X(:,1:n-1)-X(:,2:n);
      
   end