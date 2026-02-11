function [L, C] = compute_L(X, sigma)
% COMPUTE_L Compute graph Laplacian and covariance matrix for spectral clustering
% This function constructs a k-nearest neighbor graph and computes the 
% normalized graph Laplacian using Gaussian (RBF) kernel weights.
%
% Inputs:
%   X     - Data matrix (n_samples x n_features)
%   sigma - Bandwidth parameter for Gaussian kernel (controls neighborhood size)
%
% Outputs:
%   L - Graph Laplacian matrix (n_samples x n_samples), L = D - W
%   C - Covariance matrix (for potential downstream use in subspace learning)
%
% Reference: Spectral clustering based on Ng-Jordan-Weiss algorithm
    
    %% Configuration
    k = 12;  % Number of nearest neighbors for graph construction
    
    [n_samples, n_features] = size(X);
    
    %% Covariance Matrix Computation
    % Compute covariance matrix C for potential use in discriminant analysis
    % or subspace learning (e.g., LDA, CCA extensions)
    if n_features < n_samples
        % Standard covariance: features are dimensions, samples are observations
        % Result: (n_features x n_features) matrix
        C = cov(X);
    else
        % High-dimensional case: n_features >= n_samples
        % Use computational trick: XX'/(n-1) instead of X'X/(n-1)
        % This yields an (n_samples x n_samples) matrix, more efficient for eigendecomposition
        % Note: X should be centered (zero-mean) for true covariance
        C = (X * X') / n_samples;
    end
    
    %% Graph Construction: k-Nearest Neighbor Graph
    % Build sparse similarity graph using k-NN connectivity
    
    if n_samples < 4000
        % Exact k-NN search using full distance matrix (feasible for small datasets)
        % Compute pairwise squared Euclidean distances
        G = L2_distance(X', X');  % Output: (n_samples x n_samples) distance matrix
        
        % Retain only k-nearest neighbors for each point (symmetric approach)
        [~, sorted_idx] = sort(G);  % sorted_idx(i,j) = i-th nearest neighbor of j
        
        % Sparsify distance matrix: keep only k+1 smallest distances per column
        % (k+1 because first neighbor is the point itself with distance 0)
        for i = 1:n_samples
            G(i, sorted_idx((k + 2):end, i)) = 0;  % Set distant points to 0
        end
        
        % Convert to sparse format for memory efficiency
        G = sparse(double(G));
        
        % Symmetrize the graph: ensure if i is neighbor of j, j is neighbor of i
        G = (G + G') / 2;  % Average of directed graphs
        
    else
        % Approximate k-NN for large datasets using efficient search structures
        % (e.g., KD-trees, ball trees, or randomized projections)
        G = find_nn(X, k);  % Returns sparse distance matrix
    end
    
    %% Gaussian Kernel (RBF) Weight Computation
    
    % Normalize squared distances to [0, 1] for numerical stability
    % This makes sigma parameter scale-invariant
    G = G .^ 2;  % Square distances (already squared in L2_distance, but ensure)
    max_dist = max(max(G));
    if max_dist > 0
        G = G / max_dist;  % Normalize by maximum squared distance
    end
    
    % Apply Gaussian heat kernel: W_ij = exp(-||x_i - x_j||^2 / (2*sigma^2))
    % Only compute for non-zero entries (sparse optimization)
    nz_idx = (G ~= 0);
    G(nz_idx) = exp(-G(nz_idx) / (2 * sigma^2));
    
    % Result: G is now the weighted adjacency matrix W
    
    %% Graph Laplacian Construction
    
    % Compute degree matrix D (diagonal matrix of row sums)
    % D_ii = sum_j W_ij (weighted degree of node i)
    degrees = sum(G, 2);
    D = spdiags(degrees, 0, n_samples, n_samples);  % Sparse diagonal matrix
    
    % Unnormalized Laplacian: L = D - W
    % Properties: L is positive semi-definite, L*1 = 0 (constant vector is eigenvector with eigenvalue 0)
    L = D - G;
    
    %% Numerical Stability Checks
    
    % Handle potential numerical issues from kernel computation
    L(isnan(L)) = 0;   % Replace NaN (from 0/0 or inf-inf operations)
    L(isinf(L)) = 0;   % Replace Inf (from overflow in exp)
    
    % Ensure symmetry (numerical errors might break exact symmetry)
    L = (L + L') / 2;
    
end
