function d = L2_distance_subfun(a, b)
% L2_DISTANCE_SUBFUN Efficiently compute pairwise squared Euclidean distances
% This function computes the squared L2 distance between all pairs of columns
% in matrices a and b using the expansion: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a'*b
%
% Mathematical formula:
%   d(i,j) = ||a(:,i) - b(:,j)||^2 
%          = sum((a(:,i) - b(:,j)).^2)
%          = sum(a(:,i).^2) + sum(b(:,j).^2) - 2*a(:,i)'*b(:,j)
%
% Inputs:
%   a - Data matrix (m x n), each column is a data point
%   b - Data matrix (m x p), each column is a data point
%
% Output:
%   d - Distance matrix (n x p), where d(i,j) is the squared L2 distance 
%       between a(:,i) and b(:,j)
%
% Computational complexity: O(m*n*p) for matrix multiplication, 
% much faster than naive O(m*n*p) with explicit loops when vectorized
%
% Note: Returns squared distances (not square root) for efficiency
%       Use sqrt(d) if actual Euclidean distances are needed

    %% Handle edge case: 1D data
    % If input is 1D (row vector), pad with zeros to enable broadcasting
    % This ensures the matrix operations work correctly for 1D cases
    if size(a, 1) == 1
        a = [a; zeros(1, size(a, 2))];  % Pad to 2xN
        b = [b; zeros(1, size(b, 2))];  % Pad to 2xM
    end
    
    %% Compute squared norms using efficient element-wise operations
    % ||a(:,i)||^2 for each column i (1 x n vector)
    aa = sum(a .* a, 1);  % Equivalent to diag(a'*a) but much faster
    
    % ||b(:,j)||^2 for each column j (1 x p vector)
    bb = sum(b .* b, 1);  % Equivalent to diag(b'*b) but much faster
    
    %% Compute cross term: a'*b (n x p matrix)
    % ab(i,j) = a(:,i)' * b(:,j) = dot product of column i of a and column j of b
    ab = a' * b;  % Matrix multiplication: (n x m) * (m x p) = (n x p)
    
    %% Compute squared distances using broadcasting (implicit expansion)
    % d(i,j) = ||a(:,i)||^2 + ||b(:,j)||^2 - 2*a(:,i)'*b(:,j)
    %
    % repmat(aa', [1, size(bb,2)]) creates n x p matrix where row i is ||a(:,i)||^2
    % repmat(bb, [size(aa,2), 1]) creates n x p matrix where col j is ||b(:,j)||^2
    

    d = aa' + bb - 2 * ab;  % aa' is n x 1, bb is 1 x p, result is n x p
    
    % For older MATLAB versions, use explicit expansion:
    % d = repmat(aa', [1, length(bb)]) + repmat(bb, [length(aa), 1]) - 2*ab;
    
    %% Ensure real output
    % Numerical errors might produce tiny imaginary parts due to floating point
    % This clamps them to zero and ensures the result is purely real
    d = real(d);
    
    %% Numerical safeguard: clamp small negative values to zero
    % Due to floating point precision, values very close to zero might be 
    % slightly negative (e.g., -1e-16). These are mathematically zero.
    d(d < 0 & d > -1e-12) = 0;
    
end
