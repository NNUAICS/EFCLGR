function [U, beta, obj] = SimplexQP_ALM(H, D, n, c, mu, rho)
% SIMPLEXQP_ALM  Solve simplex-constrained quadratic programming using Augmented Lagrangian Method
%
%   Problem formulation:
%       minimize    x'Ax - b'x
%       subject to  x'1 = 1, x >= 0
%
%   This is solved via variable splitting and ADMM/ALM framework:
%       minimize    ||H - U + D*beta||_F^2
%       subject to  U*1 = 1, U >= 0, U = beta
%
%   Inputs:
%       H       - [n x c] data matrix (constant term)
%       D       - [n x n] diagonal or coefficient matrix
%       n       - number of samples/rows
%       c       - number of clusters/columns
%       mu      - [scalar] initial penalty parameter (mu > 0)
%       rho     - [scalar] penalty update factor (rho > 1)
%
%   Outputs:
%       U       - [n x c] primal variable (simplex-constrained)
%       beta    - [n x c] auxiliary variable (unconstrained)
%       obj     - [iter x 1] convergence history (||U - beta||_F^2)
%
%   Reference: Alternating Direction Method of Multipliers (ADMM) for
%              convex optimization with simplex constraints

% Algorithm Parameters
maxIter     = 100;          % Maximum number of iterations
tol         = 1e-6;         % Convergence tolerance for primal residual
patience    = 5;            % Early stopping patience (consecutive iterations below tol)
patienceCnt = 0;            % Counter for patience-based stopping

% Initialization
sigma   = zeros(n, c);      % Dual variable (Lagrange multiplier for U = beta)
U       = zeros(n, c);      % Primal variable with simplex constraints
beta    = initfcm(n, c);    % Initialize beta using FCM (Fuzzy C-Means) initialization

% Precompute constant term for Z update
% Z represents the linear term in the quadratic objective

% Main ADMM Iterations
obj = zeros(maxIter, 1);    % Store objective values for convergence analysis

for iter = 1:maxIter
    
    %% Step 1: Update Z (auxiliary linear term)
    % Z combines the data fidelity term and the linear contribution from beta
    Z = H + D * beta;
    
    %% Step 2: U-update (Simplex Projection)
    % Solve: min_U ||U - (beta - (sigma + Z)/mu)||_F^2 
    %        s.t. U*1 = 1, U >= 0
    % This is solved via Euclidean projection onto the probability simplex
    
    for i = 1:n
        % Compute the target point before projection
        d = beta(i, :) - (sigma(i, :) + Z(i, :)) / mu;
        % Project row i onto the probability simplex
        U(i, :) = EProjSimplex(d);
    end
    
    %% Step 3: beta-update (Unconstrained Least Squares)
    % Solve: min_beta ||D*beta - (D'*U - mu*U - sigma)||_F^2 / mu
    % Closed-form solution from first-order optimality condition
    
    beta = (-D' * U + mu * U + sigma) / mu;
    
    %% Step 4: Dual Variable Update (Ascent Step)
    % sigma_{k+1} = sigma_k + mu * (U - beta)
    % This enforces the consensus constraint U = beta
    
    sigma = sigma + mu * (U - beta);
    
    %% Step 5: Penalty Parameter Update
    % Increase mu to accelerate convergence (standard ALM heuristic)
    mu = rho * mu;
    
    %% Convergence Monitoring
    % Primal residual: measures constraint violation ||U - beta||_F^2
    obj(iter) = norm(U - beta, 'fro')^2;
    
    % Check for numerical stability
    if any(isnan(U(:)))
        error('Numerical instability detected: U contains NaN values. Check input matrices or reduce rho.');
    end
    
    % Early stopping criterion with patience
    if obj(iter) < tol
        patienceCnt = patienceCnt + 1;
        if patienceCnt >= patience
            % Converged: sufficient consecutive iterations below tolerance
            obj = obj(1:iter);  % Trim unused entries
            break;
        end
    else
        patienceCnt = 0;  % Reset patience counter if residual increases
    end
    
end

% Trim objective history if converged early
if iter < maxIter && length(obj) > iter
    obj = obj(1:iter);
end

end
