function [Y_noisy, noisy_nums, Rho] = CCN_Gen5(Y, para)
    % Generate the noisy labels
    % Y_noisy \in {1,-1}: the same dimension as Y
    % noisy_nums: The total number of injected noisy labels 
    % Rho: q*2, the column sequentially stores:{\rho_{-}, \rho_{+}}
    
    rng(0); 
    
    Y(Y == -1) = 0;
    Y_noisy = Y;
    [N, q] = size(Y);
    noisy_nums = zeros(q, 1);
    NumP = sum(Y);
    NumN = N - NumP;
    
    if ~para.sep 
        Rho = randAB(para.A, para.B, [q, 2]);
    else
        Rho(:, 1) = randAB(para.A1, para.B1, [q, 1]);
        Rho(:, 2) = randAB(para.A2, para.B2, [q, 1]);
    end
    
    TotalNumP = floor(Rho(:, 1) .* NumP');
    TotalNumN = floor(Rho(:, 2) .* NumN');
    
    % Initialize index arrays
    idx_p = cell(q, 1);
    idx_n = cell(q, 1);
    for j = 1:q
        idx_p{j} = find(Y(:, j) == 1);
        idx_n{j} = find(Y(:, j) ~= 1);
    end
    
    % Introduce noise in a vectorized manner
    for j = 1:q
        num_pos_flips = TotalNumP(j);
        num_neg_flips = TotalNumN(j);
        
        % Flip positive to negative
        if num_pos_flips > 0
            pos_indices = idx_p{j};
            rand_indices = pos_indices(randperm(length(pos_indices), min(num_pos_flips, length(pos_indices))));
            Y_noisy(rand_indices, j) = 0;
            noisy_nums(j) = noisy_nums(j) + length(rand_indices);
        end

        % Flip negative to positive
        if num_neg_flips > 0
            neg_indices = idx_n{j};
            rand_indices = neg_indices(randperm(length(neg_indices), min(num_neg_flips, length(neg_indices))));
            Y_noisy(rand_indices, j) = 1;
            noisy_nums(j) = noisy_nums(j) + length(rand_indices);
        end
    end

    Y_noisy = Y_noisy';
    Y_noisy(Y_noisy == 0) = -1;
end

function R = randAB(A, B, sz)
    R = A + (B - A) .* rand(sz);
end
