function [L,C] = compute_L(X,sigma)
     k=12;
    %% Calculate the covariance matrix C
    if size(X, 2) < size(X, 1)
        C = cov(X);
    else
        C = (1 / size(X, 1)) * (X * X');        % if N>D, we better use this matrix for the eigendecomposition
    end
    %% Calculate the Laplacian matrix L
    if size(X, 1) < 4000
        G = L2_distance(X', X');
        % Compute neighbourhood graph
        [~, ind] = sort(G); 
        for i=1:size(G, 1)
            G(i, ind((2 + k):end, i)) = 0; 
        end
        G = sparse(double(G));
        G = (G + G')/2;
        %********G = max(G, G');             % Make sure distance matrix is symmetric
    else
        G = find_nn(X, k);
    end
    G = G .^ 2;
	G = G ./ max(max(G));
    
    % Compute weights (W = G)
    %disp('Computing weight matrices...');
    
    % Compute Gaussian kernel (heat kernel-based weights)
    G(G ~= 0) = exp(-G(G ~= 0) / (2 * sigma ^ 2));
        
    % Construct diagonal weight matrix
    D = diag(sum(G, 2));
    L = D - G;
    L(isnan(L)) = 0; 
    L(isinf(L)) = 0; 
end