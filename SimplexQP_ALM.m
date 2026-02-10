function [U,beta,obj] = SimplexQP_ALM(H,D,n,c,mu,rho)
% solve:
% min     x'Ax - b'x
% s.t.    x'1 = 1, x >= 0
% paras:
% mu    - mu > 0
% beta  - 1 < beta < 2
%
NITER = 500;
THRESHOLD = 1e-6;
sigma = ones(n,c);
A = zeros(n,c);
cnt = 0;
beta = initfcm(n,c);

for iter = 1:NITER
    Z = H+D*beta;
    for i = 1:n
        dd(i,:) = beta(i,:) -(sigma(i,:)+Z(i,:))/mu;
        A(i,:) =  EProjSimplex(dd(i,:));
    end
    U = A;
   
    beta=(-D'*U+mu*U+sigma)/mu;
    Beta{iter}=beta;
    sigma = sigma + mu*(U - beta);
    mu = rho*mu;
    obj(iter)=norm(U-beta,'fro').^2;
    
    if obj(iter) < THRESHOLD
        if cnt >= 5
        %disp(['U', num2str(iter)]);
            break;
        else
            cnt = cnt + 1;
        end
    else
        cnt = 0;
    end
    if any(isnan(U(:)))
        error('U 计算出 NaN，请检查数值稳定性！');
    end
end


end