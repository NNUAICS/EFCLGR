function [x] = EProjSimplex(v, k)
% EPROJSIMPLEX Euclidean projection onto the probability simplex
% This function computes the projection of a vector v onto the standard 
% probability simplex using an efficient O(n log n) algorithm based on 
% sorting or an O(n) expected time algorithm based on the bisection method.
%
% Optimization Problem:
%   min_x  (1/2) * ||x - v||^2
%   subject to: x >= 0, 1'*x = k (typically k=1 for probability simplex)
%
% This is a fundamental subroutine in many machine learning algorithms,
% including projected gradient descent for problems with simplex constraints.
%
% Inputs:
%   v - Input vector (n x 1) to be projected
%   k - Simplex constraint parameter (default: 1)
%       * k=1: probability simplex (sum(x) = 1, x >= 0)
%       * k>0: scaled simplex (sum(x) = k, x >= 0)
%
% Output:
%   x - Projected vector (n x 1) satisfying x >= 0 and sum(x) = k
%
% Algorithm: Bisection method on the dual variable (Lagrange multiplier)
% Reference: Chen and Ye (2011), "Projection onto a simplex"
    
    %% Input Handling
    if nargin < 2
        k = 1;  % Default: standard probability simplex
    end
    
    %% Initialization
    ft = 1;                     % Iteration counter for safeguard
    n = length(v);              % Dimension of the problem
    
    % Initial shift: center v and adjust for constraint sum(x) = k
    % v0 = v - mean(v) + k/n ensures that sum(v0) = k
    % This provides a good starting point for the bisection method
    v0 = v - mean(v) + k/n;
    
    %% Check Feasibility of Initial Point
    % If v0 >= 0 (all elements non-negative), then v0 is already the solution
    % because it satisfies both constraints: sum(v0) = k and v0 >= 0
    vmin = min(v0);
    
    if vmin >= 0
        % v0 is feasible and optimal (no projection needed)
        x = v0;
        
    else
        % v0 violates non-negativity constraint
        % Need to find optimal Lagrange multiplier lambda* such that
        % x = max(v0 - lambda*, 0) and sum(x) = k
        
        % Initialize bisection method
        f = 1;                  % Constraint violation (will drive to 0)
        lambda_m = 0;           % Initial guess for Lagrange multiplier
        
        % Bisection/Secant method to find root of f(lambda) = sum(max(v0-lambda,0)) - k
        while abs(f) > 1e-10    % Convergence tolerance
            
            % Compute shifted vector
            v1 = v0 - lambda_m;
            
            % Identify active set: indices where v1 > 0 (x will be non-zero)
            posidx = (v1 > 0);
            npos = sum(posidx);  % Number of positive elements
            
            % Compute gradient of dual function: g = -npos = -d/dlambda sum(max(v1,0))
            % This is the derivative of the sum of positive parts with respect to lambda
            g = -npos;
            
            % Compute constraint violation: f = sum(x) - k where x = max(v1, 0)
            % We want f = 0 (satisfies equality constraint)
            f = sum(v1(posidx)) - k;
            
            % Newton/bisection update: lambda_{new} = lambda - f/g
            % This is a root-finding step for f(lambda) = 0
            lambda_m = lambda_m - f/g;
            
            % Safeguard: limit iterations to prevent infinite loops
            ft = ft + 1;
            if ft > 100
                % Fallback: return best estimate so far
                x = max(v1, 0);
                break;
            end
        end
        
        % Final solution: apply soft-thresholding with optimal lambda
        x = max(v1, 0);
    end
end
