clc;
clear;
warning('off');

% Configuration Section
% Dataset and tool configuration
datasets_name = {'isolet_uni'};
tools_name = {'EFCLGR'};

% Base paths for data and results
base_data_path = 'D:\Matlab\DM\data\data\';
base_result_path = 'D:\Matlab\DM\LPPFCM\EFCLGR\result\';

%% Parameter Grid Definition
% Hyperparameter search space for EFCLGR algorithm
sigma_vals = [0.1, 1, 10];      % Gaussian kernel bandwidth parameters
lambda_vals = [0.1, 1, 10];     % Regularization parameter (corrected spelling from 'lamda')
gamma_vals = [0.1, 1, 10];      % Graph regularization weight
theta_vals = [0.1, 1, 10];      % Constraint parameter
mu_vals = 100;                  % Penalty parameter for ALM
rho_vals = 1.01;                % Augmented Lagrangian multiplier update rate
n_iterations = 100;              % Number 

%% Main Processing Loop
for ds = 1:length(datasets_name)
    dataset = datasets_name{ds};
    file = fullfile(base_data_path, [dataset, '.mat']);
    
    % Check if data file exists to prevent runtime errors
    if ~exist(file, 'file')
        warning('Data file does not exist: %s', file);
        continue;
    end
    
    load(file);  % Load dataset variables (X: features, Y: labels)
    
    %% Preprocessing: PCA Dimensionality Reduction
    % Apply PCA once before the dimension loop to avoid redundant computation
    if size(X, 2) > 100
        X = double(X);  % Ensure double precision for numerical stability
        [~, data] = pca(X, 'NumComponents', 100);  % Reduce to 100 dimensions
    else
        data = X;
    end
    
    [n, ~] = size(data);
    strat_dim = 10;     % Starting dimension for subspace clustering
    step = 10;          % Step size for dimension increment
    end_dim = 100;      % Maximum dimension to evaluate
    
    %% Tool Selection Loop
    for drt = 1:length(tools_name)
        tool = tools_name{drt};
        filename = fullfile(base_result_path, sprintf('%s_%s.xlsx', dataset, tool));
        
        % Skip if result file already exists to enable resume capability
        if exist(filename, 'file')
            continue;
        end
        
        class = length(unique(Y));  % Number of clusters (classes)
        result = [];                % Initialize results accumulator
        
        %% Dimension Sweep Loop
        for dim = strat_dim:step:end_dim
            fprintf('Processing dim = %d\n', dim);
            
            if strcmp(tool, "EFCLGR")
                %% Sigma Loop: Precompute Laplacian matrices
                % L: Graph Laplacian, C: Constraint matrix
                for sigma = sigma_vals
                    [L, C] = compute_L(data, sigma);  % Expensive operation, done per sigma
                    
                    %% Vectorized Parameter Combination
                    % Generate all combinations of hyperparameters using Cartesian product
                    % This replaces 4 nested loops with a single matrix operation
                    param_grid = combvec(lambda_vals, gamma_vals, theta_vals, mu_vals, rho_vals)';
                    n_comb = size(param_grid, 1);
                    
                    %% Hyperparameter Grid Search
                    for i = 1:n_comb
                        lambda = param_grid(i, 1);
                        gamma = param_grid(i, 2);
                        theta = param_grid(i, 3);
                        mu = param_grid(i, 4);
                        rho = param_grid(i, 5);
                        
                        % Pre-allocate memory for performance metrics
                        ACCs = zeros(n_iterations, 1);      % Accuracy scores
                        NMIs = zeros(n_iterations, 1);      % Normalized Mutual Information
                        PURITYs = zeros(n_iterations, 1);   % Clustering purity scores
                        
                        %% Monte Carlo Simulation
                        % Run multiple iterations to account for algorithm randomness
                        for a = 1:n_iterations
                            % Execute EFCLGR clustering algorithm
                            U = EFCLGR(data, L, C, class, dim, lambda, gamma, theta, mu, rho);
                            
                            % Extract cluster assignments (hard clustering)
                            [~, index] = max(U, [], 2);
                            
                            % Evaluate clustering performance against ground truth
                            [ACC, MIhat, Purity] = ClusteringMeasure(Y', index');
                            
                            % Store metrics for statistical analysis
                            ACCs(a) = ACC;
                            NMIs(a) = MIhat;
                            PURITYs(a) = Purity;
                        end
                        
                        % Compute mean and standard deviation across iterations
                        [mean_m, std_m] = ud_measure(ACCs, NMIs, PURITYs);
                        
                        % Append results: [mean_metrics, std_metrics, hyperparameters, dimension]
                        result = [result; mean_m, std_m, sigma, lambda, gamma, theta, mu, rho, dim]; %#ok<AGROW>
                    end
                end
            end
        end
        
        %% Result Persistence
        % Write results to Excel file if any computations were performed
        if ~isempty(result)
            writematrix(result, filename);
        end
    end
end

%% Utility Functions

function [mean_measure, std_measure] = ud_measure(ACC, MIhat, Purity)
% UD_MEASURE Calculate mean and standard deviation of clustering metrics
% Input:
%   ACC    - Accuracy values across iterations (column vector)
%   MIhat  - NMI values across iterations (column vector)
%   Purity - Purity values across iterations (column vector)
% Output:
%   mean_measure - [mean_ACC, mean_NMI, mean_Purity]
%   std_measure  - [std_ACC, std_NMI, std_Purity]
    
    measure = [ACC, MIhat, Purity];
    mean_measure = mean(measure, 1);           % Mean along rows
    std_measure = std(measure, 0, 1);          % Sample std along rows (N-1 normalization)
end

function [mean_measure, std_measure] = udran_measure(data, target, num_samp, iters)
% UDRAN_MEASURE Evaluate clustering via random sampling and 1-NN classification
% Input:
%   data     - Feature matrix (n_samples x n_features)
%   target   - Ground truth labels (n_samples x 1)
%   num_samp - Number of samples per class for training
%   iters    - Number of random sampling iterations
% Output:
%   mean_measure - [mean_accuracy, mean_macro_f1, mean_micro_f1]
%   std_measure  - [std_accuracy, std_macro_f1, std_micro_f1]
    
    measure = zeros(iters, 3);  % Pre-allocate: [accuracy, macro_f1, micro_f1]
    
    for count = 1:iters
        % Stratified random sampling to ensure class balance
        [~, ~, idx_train] = Random_sampling(target, num_samp, 'class');
        idx_test = setdiff(1:length(target), idx_train);
        
        % Train 1-Nearest Neighbor classifier
        mdl = fitcknn(data(idx_train, :), target(idx_train), ...
                      'NumNeighbors', 1, 'Distance', 'euclidean');
        pred_label = predict(mdl, data(idx_test, :));
        
        % Evaluate classification performance
        out = classification_evaluation(target(idx_test)', pred_label');
        measure(count, :) = [out.avgAccuracy, out.fscoreMacro, out.fscoreMicro];
    end
    
    mean_measure = mean(measure, 1);
    std_measure = std(measure, 0, 1);
end

function [mean_measure, std_measure] = new_measure(data, class, target, num_count)
% NEW_MEASURE Evaluate k-means clustering stability over multiple runs
% Input:
%   data      - Feature matrix
%   class     - Number of clusters (k)
%   target    - Ground truth labels
%   num_count - Number of k-means runs with different initializations
% Output:
%   mean_measure - [mean_ACC, mean_NMI, mean_Purity]
%   std_measure  - [std_ACC, std_NMI, std_Purity]
    
    measure = zeros(num_count, 3);  % Pre-allocate performance matrix
    
    for count = 1:num_count
        % Run k-means with random initialization
        index = kmeans(data, class);
        
        % Compare with ground truth
        [ACC, MIhat, Purity] = ClusteringMeasure(target', index');
        measure(count, :) = [ACC, MIhat, Purity];
    end
    
    mean_measure = mean(measure, 1);
    std_measure = std(measure, 0, 1);
end

