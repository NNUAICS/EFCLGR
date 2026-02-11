function [U] = EFCLGR(X, L, C, label, wd, lambda, gamma, theta, mu, rho)
% 
% This function implements an iterative optimization algorithm that combines
% Locality Preserving Projections (LPP) with Fuzzy C-Means (FCM) clustering.
%
% Optimization Problem:
%   min_{P,U}  ||P'*x_i - b_k||^2 + theta*s_{ij}*||P'*x_i - P'*x_j||^2 
%              + lambda*Tr(U'*L*U) + gamma*Tr(U'*11'*U)
%   subject to: P'*S*P = I, U >= 0, U*1 = 1
%
% where:
%   P  - Projection matrix (d x wd), projects data to wd-dimensional subspace
%   U  - Fuzzy membership matrix (n x c), U_ik indicates membership of point i to cluster k
%   F  - Cluster centers in original space (d x c)
%   L  - Graph Laplacian matrix (n x n), encodes data manifold structure
%   C  - Constraint matrix for generalized eigenvalue problem
%
% Inputs:
%   X     - Data matrix (n x d), n samples, d features
%   L     - Graph Laplacian matrix (n x n) from compute_L()
%   C     - Constraint/covariance matrix (d x d) for generalized eigendecomposition
%   label - Number of clusters c (scalar)
%   wd    - Target dimensionality after projection (scalar)
%   lambda- Weight for Laplacian regularization term Tr(U'*L*U)
%   gamma - Weight for fuzzy entropy regularization Tr(U'*11'*U)
%   theta - Weight for locality preserving term in projection
%   mu    - Initial penalty parameter for Augmented Lagrangian Method (ALM)
%   rho   - Augmentation factor for ALM (typically 1.01-1.1)
%
% Output:
%   U     - Optimized fuzzy membership matrix (n x c)
%
% Reference: Integrated dimensionality reduction and clustering with graph regularization

    %% Algorithm Configuration
    maxgen = 100;        % Maximum number of outer iterations for alternating optimization
    tol = 1e-6;         % Convergence tolerance for objective function change
    maxit = 0;          % ALM inner iteration counter (initialized to 0)
    
    %% Initialization: Fuzzy Membership Matrix U
    c = label;          % Number of clusters
    [n, d] = size(X);   % n: number of samples, d: number of features
    
    % Initialize U with random values in [0, 1] and normalize rows to sum to 1
    % This satisfies the probabilistic constraint: sum_k U_ik = 1 for all i
    U = rand(n, c);                         % Random initialization
    row_sum = sum(U, 2);                    % Compute row sums (n x 1)
    U = U ./ repmat(row_sum, 1, c);         % Normalize: U_ik = U_ik / sum_j(U_ij)
    
    % Precompute constant matrix A for membership update
    % A combines graph Laplacian regularization (lambda*L) and 
    % fuzzy entropy regularization (gamma*1*1')
    A = lambda * L + gamma * ones(n, 1) * ones(1, n);  % (n x n) matrix
    
    %% Alternating Optimization Loop
    % Iteratively optimize projection P and membership U until convergence
    
    prev_J = inf;       % Previous objective value for convergence check
    
    for i = 1:maxgen
        %% Step 1: Fix U, Optimize Projection P and Centers F
        
        % Compute cluster centers F in original space (weighted by memberships)
        % F_j = (sum_i U_ij * x_i) / (sum_i U_ij)  for each cluster j
        sum_U = sum(U, 1);                      % Column sums: cluster weights (1 x c)
        sum_U(sum_U < eps) = eps;               % Avoid division by zero
        F = (X' * U) ./ repmat(sum_U, d, 1);    % Cluster centers (d x c)
        
        % Construct scatter matrices for generalized eigenvalue problem
        dn = diag(sum(U, 2));                   % Diagonal: row sums of U (n x n)
        dc = diag(sum(U, 1));                   % Diagonal: column sums of U (c x c)
        
        % Compute M matrix: within-cluster scatter in projected space
        % M = X'*Dn*X - 2*X'*U*F' + F*Dc*F'
        % This represents the weighted variance of data around cluster centers
        M = X' * dn * X - 2 * X' * U * F' + F * dc * F';
        M = (M + M') / 2;                       % Symmetrize for numerical stability
        M(isnan(M)) = 0;                        % Clean numerical errors
        M(isinf(M)) = 0;
        
        % Add locality preserving term: theta * X' * L * X
        % This ensures the projection preserves local neighborhood structure
        M = M + theta * X' * L * X;             % Final scatter matrix (d x d)
        
        % Solve generalized eigenvalue problem: M*V = C*V*B
        % Find projection P that minimizes scatter while satisfying P'*C*P = I
        [V, B] = eig(M, C);                     % V: eigenvectors, B: eigenvalues
        B(isnan(B)) = 0;                        % Clean numerical errors
        
        % Sort eigenvalues in ascending order and select smallest wd eigenvectors
        % These correspond to directions of minimum variance (optimal for clustering)
        [~, ind] = sort(diag(B));               % Sort eigenvalues
        P = V(:, ind(1:wd));                    % Select wd smallest eigenvectors (d x wd)
        
        %% Step 2: Fix P and F, Optimize Membership U
        
        % Project data to low-dimensional subspace: Y = X * P (n x wd)
        Y = X * P;
        
        % Project cluster centers: PF = P' * F (wd x c)
        PF = P' * F;
        
        % Compute squared distances between projected points and centers
        % H_ik = ||y_i - pf_k||^2, where y_i is projected point i, pf_k is projected center k
        H = L2_distance_subfun(Y', PF);         % Distance matrix (n x c)
        
        % Solve constrained quadratic program for U using ALM
        % min_U Tr(U'*H*U) + Tr(U'*A*U)  s.t. U >= 0, U*1 = 1
        [U, beta, obj] = SimplexQP_ALM(H, A, n, c, mu, rho);
        
        %% Step 3: Compute Objective Function and Check Convergence
        
        % Total objective: reconstruction error + graph regularization + entropy regularization
        % J = Tr(P'*M*P) + lambda*Tr(U'*L*U) + gamma*Tr(U'*11'*U)
        J(i) = trace(P' * M * P) + ...
               lambda * trace(U' * L * U) + ...
               gamma * trace(U' * ones(n, 1) * ones(1, n) * U);
        
        % Check ALM iteration limit (emergency stop)
        if maxit > 100
            break;
        end
        
        % Check convergence: stop if objective change is below tolerance
        % Note: Original code had break uncommented, effectively running only 1 iteration
        % The following implements proper convergence checking:
        if abs(J(i) - prev_J) < tol
            break;
        end
        
        prev_J = J(i);                          % Update previous objective value
    end
    
end
