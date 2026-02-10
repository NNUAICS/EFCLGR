function [U] = LPPFCM(X,L,C,label,wd,lamda,gamma,theta,mu,rho)
%% Problem
%  
%  min ||P'*x_i-b_k||^2+theta*s_{ij}*||P'*x_i-P'*x_j||^2 +lamda*Tr(U'LU)+gamma*Tr(U'11'U)
%  s.t. P'SP=I,U>=0,U1=1
%
%  P is the projection matrix
%  U is the membership matrix
%  F is the cluster center in the original space.
    %% Initial objective function
    clear J;
    maxgen =30;
    %k=12;
    tol = 1e-5; 
    t=0;
    maxit=0;
    %% Randomly generate U with values in the range [0, 1].
    c=label; 
    [n,d] =size(X);
    U=rand(n,c);
    row_sum=sum(U, 2); 
    U=U./repmat(row_sum,1,c); 

    %% Iterative solving
    prev_J = inf; 
    A=lamda*L+gamma*ones(n, 1)*ones(1, n);
    for i = 1:maxgen
        % Fix U and solve for F and P
        sum_U = sum(U,1);
        sum_U(sum_U < eps) = eps;
        F=X'*U./(repmat(sum_U,d,1));
        dn=diag(sum(U,2));
        dc=diag(sum(U,1));
        M=X'*dn*X-2*X'*U*F'+F*dc*F';
        M=(M+M')/2;
        M(isnan(M)) = 0;
        M(isinf(M)) = 0;
        M=(M+theta*X'*L*X);

        [V,B]=eig(M,C);
        B(isnan(B)) = 0;
        [~, ind] = sort(diag(B));
        P = V(:,ind(1:wd)); 
        %P(isnan(P)) = 0;
        % Fix F and P, and solve for U
        Y=X*P;
        PF=P'*F;
        H =L2_distance_subfun(Y',PF);
        [U,beta,obj] = SimplexQP_ALM(H,A,n,c,mu,rho);
%         AA=trace(Y'*L*Y);
%         BB=trace(U'*L*U);
%         CC=trace(U'*ones(n,1)*ones(1,n)*U);
         
        J(i) = trace(P'*M*P)+lamda*trace(U'*L*U)+gamma*trace(U'*ones(n,1)*ones(1,n)*U);
        if maxit>100
         break;
        end
%          if abs(J(i) - prev_J) < tol
            break;
         end
        prev_J = J(i); 
%     end 
   
end
