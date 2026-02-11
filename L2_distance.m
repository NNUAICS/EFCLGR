function d = L2_distance(a, b)
% L2_DISTANCE - computes Euclidean distance matrix
%
% E = L2_distance(A,B)
%
%    A - (DxM) matrix 
%    B - (DxN) matrix
% 
% Returns:
%    E - (MxN) Euclidean distances between vectors in A and B
%
%
% Description : 
%    This fully vectorized (VERY FAST!) m-file computes the 
%    Euclidean distance between two vectors by:
%
%                 ||A-B|| = sqrt ( ||A||^2 + ||B||^2 - 2*A.B )
%
% Example : 
%    A = rand(400,100); B = rand(400,200);
%    d = distance(A,B);



    if nargin < 2
       error('Not enough input arguments');
    end
    if size(a, 1) ~= size(b, 1)
        error('A and B should be of same dimensionality');
    end
    if ~isreal(a) || ~isreal(b)
        warning('Computing distance table using imaginary inputs. Results may be off.'); 
    end

    % Padd zeros if necessray
    if size(a, 1) == 1
        a = [a; zeros(1, size(a, 2))]; 
        b = [b; zeros(1, size(b, 2))]; 
    end

    % Compute distance table
    d = sqrt(bsxfun(@plus, sum(a .* a)', bsxfun(@minus, sum(b .* b), 2 * a' * b)));

    % Make sure result is real
    d = real(d);

