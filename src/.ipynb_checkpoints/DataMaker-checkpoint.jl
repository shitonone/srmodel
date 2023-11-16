module DataMaker

using Statistics: mean
using LinearAlgebra: svd, Diagonal

export make_data

function make_data(;
        data, t_list, k_list, test_size, pc_param
)
    max_t = maximum(t_list)
    max_k = maximum(k_list)
    
    targets = cat([_pct_change(data, max_t, max_k, t, nothing) for t in t_list]..., dims=3) 
    # = (n_samples, n_coins, length(t_list))
    targets = permutedims(targets, [2, 3, 1]) 
    # = (n_coins, length(t_list), n_samples)
    train_targets = targets[:, :, 1:end-test_size]
    test_targets = targets[:, :, end-test_size+1:end]

    raw_returns = transpose(_pct_change(data, max_t, max_k, nothing, 1)) 
    # = (n_coins, n_samples)
    raw_returns_train = raw_returns[:, 1:end-test_size]
    raw_returns_test = raw_returns[:, end-test_size+1:end]
    
    r_ = cat([_pct_change(data, max_t, max_k, nothing, k) for k in k_list]..., dims=3) 
    # = (n_samples, n_coins, len(k_list))
    r_train = r_[1:end-test_size, :, :]
    r_test = r_[end-test_size+1:end, :, :]
    A_list = [_calc_A(r_train[:, :, i], pc_param) for i in 1:length(k_list)]

    train_features = cat([transpose(A_list[i] * transpose(r_train[:, :, i])) for i in 1:length(k_list)]..., dims=3) 
    # = (n_samples - test_size, n_coins, len(k_list))
    test_features = cat([transpose(A_list[i] * transpose(r_test[:, :, i])) for i in 1:length(k_list)]..., dims=3) 
    # = (test_size, n_coins, len(k_list))
    train_features = permutedims(train_features, [2, 1, 3])
    # = (n_coins, n_samples - test_size, len(k_list))
    test_features = permutedims(test_features, [2, 1, 3])
    # = (n_coins, test_size, len(k_list))
    
    return raw_returns_train, train_features, train_targets, raw_returns_test, test_features, test_targets
end

function _pct_change(array, max_t, max_k, t, k)
    if k == nothing
        return array[max_k+t+1:end-max_t+t, :] ./ array[max_k+1:end-max_t, :] .- 1
    elseif t == nothing
        return array[max_k+1:end-max_t, :] ./ array[max_k-k+1:end-max_t-k, :] .- 1
    end
end

function _calc_A(data, pc_param)
    X = transpose(data .- mean(data, dims=1))
    V, S, U = svd(X)
    diag = _calc_diag(S, pc_param)
    A = V * diag * transpose(V)
    return A
end

function _calc_diag(S, pc_param)
    length_S = length(S)
    sum_S = sum(S)
    pc_limit = sum_S * pc_param
    cumulative_sum = 0
    num_remove = 0
    for i in 1:length_S
        cumulative_sum += S[i]
        if cumulative_sum >= pc_limit
            break
        end
        num_remove += 1
    end
    diag = Diagonal(vcat(zeros(num_remove), ones(length_S - num_remove)))
    return diag
end

end