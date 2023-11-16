module DecisionTreeAgent

using Statistics: mean, stdm, quantile

export construct_agent_tree, predict_positions

mutable struct Node
    root::Union{Bool, Nothing}
    depth::Union{Int64, Nothing}
    label::Union{Int64, Nothing}
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    operator::Union{Function, Nothing}
    feature::Union{Int64, Nothing}
    threshold::Union{Float64, Nothing}
end

function Node(;
        root=nothing, depth=nothing,
        label=nothing, left=nothing, right=nothing,
        operator=nothing, feature=nothing, threshold=nothing
)
    return Node(
        root, depth,
        label, left, right, 
        operator, feature, threshold
    )
end


function construct_agent_tree(;
        raw_returns::Vector{Float64}, features::Matrix{Float64}, 
        fee::Float64, max_depth::Int64, n_thresholds::Int64,
        cum_return_thresholds::Vector{Float64}, position_len_thresholds::Vector{Float64},
        return_score::Bool=false
)
    
    n_features::Int64 = size(features)[2]
    threshold_list::Vector{Vector{Float64}} = vcat(
        _calc_thresholds(features, n_thresholds),
        [
            Float64[-1, 0, 1],
            cum_return_thresholds,
            position_len_thresholds
        ]
    )
    operators_list::Vector{Function} = [
        (x, y) -> x > y,
        (x, y) -> x == y
    ]
    operator_indices::Vector{Int64} = vcat(ones(Int64, n_features), Int64[2, 1, 1])
    leaf_order::Vector{Vector{Int64}} = [
        [-1, 0], [0, 1], [1, -1], [-1, 1], [1, 0], [0, -1]
    ]
    root::Node = Node(root=true, depth=0)
    stack::Vector{Node} = [root]
    current_best_score::Float64 = 0

    while length(stack) > 0
        prev_best::Float64 = current_best_score
        node_ind::Int64 = rand(1:length(stack))
        node = stack[node_ind]
        current_label = node.label
        node.label = nothing
        feature_indices::Vector{Int64} = collect(1:(n_features + 3))
        feature_indices_root::Vector{Int64} = collect(1:n_features)

        while true
            if node.root == true && length(feature_indices_root) > 0
                ind::Int64 = rand(1:length(feature_indices_root))
                feature_ind::Int64 = feature_indices_root[ind]
                popat!(feature_indices_root, ind)
                
            elseif node.root != true && length(feature_indices) > 0
                ind = rand(1:length(feature_indices))
                feature_ind = feature_indices[ind]
                popat!(feature_indices, ind)
                
            else break
            end

            node.feature = feature_ind
            node.operator = operators_list[operator_indices[feature_ind]]
            thresholds::Vector{Float64} = threshold_list[feature_ind]
            best_score::Float64 = current_best_score
            best_params::Dict{String, Any} = Dict(
                "thresh" => 0.0,
                "order" => [0, 0]
            )

            for thresh::Float64 in thresholds
                node.threshold = thresh

                for order::Vector{Int64} in leaf_order
                    node.left = Node(label=order[1])
                    node.right = Node(label=order[2])

                    r_series::Vector{Float64} = predict_positions(root, features, raw_returns, fee)[:, 4] 
                    r_mean::Float64 = mean(r_series)
                    score::Float64 = r_mean / stdm(r_series, r_mean)

                    if score > best_score
                        best_score = score
                        best_params["thresh"] = thresh
                        best_params["order"] = order
                    end
                end
            end

            if best_score > current_best_score
                current_best_score = best_score
                node.threshold = best_params["thresh"]
                node.left = Node(
                    label = best_params["order"][1],
                    depth = node.depth + 1
                )
                node.right = Node(
                    label = best_params["order"][2],
                    depth = node.depth + 1
                )

                if node.depth + 1 < max_depth
                    append!(stack, [node.left, node.right])
                end
                
                break
            end
        end

        if prev_best == current_best_score
            node.label = current_label
            node.left = nothing
            node.right = nothing
            node.operator = nothing
            node.feature = nothing
        end

        popat!(stack, node_ind)
    end

    if return_score == true
        return root, current_best_score
    else
        return root
    end
end

function _calc_thresholds(features, n_thresholds)
    n_features = size(features)[2]
    thresholds = []
    for i in 1:n_features
        thresh_ = quantile(features[:, i], [j / (n_thresholds + 1)　for j in 1:n_thresholds])
        push!(thresholds, thresh_)
    end
    
    return thresholds
end

function predict_positions(root, features, raw_returns, fee)
    n_samples = size(features)[1]
    positions = zeros(Float64, n_samples, 4)
    # positions[:, 1] : position series
    # positoins[:, 2] : position length
    # positions[:, 3] : cumlative return
    # positions[:, 4] : returns

    for i in 1:(n_samples - 1)
        data_i = vcat(features[i, :], positions[i, 1:3])
        positions[i+1, 1] = _one_position(data_i, root)
        positions[i+1, 2], positions[i+1, 3], positions[i+1, 4] = _calc_positions(i, raw_returns, positions, fee)
    end

    return positions
end

function _one_position(data_i, node)
    if node.label != nothing
        return node.label
        
    elseif node.operator(data_i[node.feature], node.threshold)
        return _one_position(data_i, node.right)

    else 
        return _one_position(data_i, node.left)
    end
end

function _calc_positions(i, raw_returns, positions, fee)
    if positions[i+1, 1] != positions[i, 1]
        position_len::Int64 = 0

        if positions[i+1, 1] == 0
            return position_len, 0, 0
            
        elseif positions[i+1, 1] == -1
            r_ = - raw_returns[i+1]
            return position_len, r_ - fee, r_ - fee

        elseif positions[i+1, 1] == 1
            r_ = raw_returns[i+1]
            return position_len, r_ - fee, r_ - fee
        end

    else
        position_len = positions[i, 2] + 1

        if positions[i+1, 1] == 0
            return position_len, 0, 0
            
        elseif positions[i+1, 1] == -1
            r_ = - raw_returns[i+1]
            return position_len, (1 + positions[i, 3]) * (1 + r_) - 1, r_
            
        elseif positions[i+1, 1] == 1
            r_ = raw_returns[i+1]
            return position_len, (1 + positions[i, 3]) * (1 + r_) - 1, r_
        end
    end
end

end