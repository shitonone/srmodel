module DecisionTreePredictor

using Statistics: mean, std, stdm, quantile
using Combinatorics: combinations

export construct_predictor_tree, predict_labels, construct_ensemble_tree, predict_ensemble_tree

mutable struct Node
    root::Union{Bool, Nothing}
    depth::Union{Int64, Nothing}
    label::Union{Int64, Nothing}
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    k::Union{Int64, Nothing}
    l::Union{Int64, Nothing}
    O_ind::Union{Int64, Nothing}
    thresh::Union{Float64, Nothing}
    indices::Union{SubArray{Int64}, Vector{Int64}, Nothing}
end

function Node(;
    root=nothing, depth=nothing,
    label=nothing, left=nothing, right=nothing,
    k=nothing, l=nothing, O_ind=nothing, 
    thresh=nothing, indices=nothing
)
    return Node(
        root, depth,
        label, left, right,
        k, l, O_ind, 
        thresh, indices == nothing ? Vector{Int}() : indices
    )
end

mutable struct Params
    k::Int64
    l::Int64
    O_ind::Int64
    thresh::Float64
    order::Vector{Int64}
    left_indices::Union{SubArray{Int64}, Vector{Int64}}
    right_indices::Union{SubArray{Int64}, Vector{Int64}}
end

function Params(;
        k=0, l=0, O_ind=0, 
        thresh=0, order=[0],
        left_indices=[0], right_indices=[0]
)
    return Params(
        k, l, O_ind,
        thresh, order,
        left_indices, right_indices
    )
end

function construct_ensemble_tree(;
        ensemble_params::Dict{Int64, Int64},
        features::SubArray{Float64}, targets::SubArray{Float64}, 
        min_samples::Int64, n_thresholds::Int64
)
    depth_list::Vector{Int64} = collect(keys(ensemble_params))
    n_models_list::Vector{Int64} = collect(values(ensemble_params))
    root_list::Vector{Node} = []

    for i in 1:length(depth_list)
        depth = depth_list[i]
        n_models = n_models_list[i]

        for j in 1:n_models
            root = construct_predictor_tree(
                features = features, targets = targets,
                max_depth = depth, min_samples = min_samples, n_thresholds = n_thresholds
            )
            push!(root_list, root)
            
        end
    end
    return root_list
end

function predict_ensemble_tree(;
        root_list::Vector{Node}, features, return_labels=false
)
    all_labels::Matrix{Int64} = zeros(Int64, size(features)[1], length(root_list))

    for i in 1:length(root_list)
        root::Node = root_list[i]
        labels = predict_labels(
            node = root, features = features
        )
        all_labels[:, i] = labels
    end

    if return_labels == true
        return all_labels
    else
        labels_mean = vec(mean(all_labels, dims=2))
        labels_std = vec(std(all_labels, dims=2))
        return labels_mean, labels_std
    end
end
    
function construct_predictor_tree(;
        features::SubArray{Float64}, targets::SubArray{Float64}, 
        max_depth::Int64, min_samples::Int64, n_thresholds::Int64
)
    n_samples::Int64 = size(features)[1]
    n_features::Int64 = size(features)[2]
    kl_list::Vector{Int64} = collect(1:n_features)
    leaf_order::Vector{Vector{Int64}} = [
        [-1, 0], [0, 1], [1, -1], [-1, 1], [1, 0], [0, -1]
    ]
    
    root::Node = Node(root=true, depth=0, indices=collect(1:n_samples))
    stack::Vector{Node} = [root]
    init_score::Float64 = mean(targets) / std(targets)
    current_best_score::Float64 = max(init_score, -init_score)
    all_labels::Vector{Int64} = zeros(Int64, n_samples)

    while length(stack) > 0
        node_ind::Int64 = rand(1:length(stack))
        node = stack[node_ind]
        node_indices = node.indices
        node_features::SubArray{Float64} = @view features[node_indices, :]
        node_targets::SubArray{Float64} = @view targets[node_indices]
        all_labels_::Vector{Int64} = copy(all_labels)
        kl_combinations::Vector{Vector{Int64}} = collect(combinations(kl_list, 2))

        while true
            if length(kl_combinations) > 0
                kl_ind::Int64 = rand(collect(1:1:length(kl_combinations)))
            else
                break
            end
            k::Int64, l::Int64 = kl_combinations[kl_ind]
            node_k_features::SubArray{Float64} = @view node_features[:, k]
            node_l_features::SubArray{Float64} = @view node_features[:, l]
            popat!(kl_combinations, kl_ind)
            best_score::Float64 = current_best_score
            best_labels::Vector{Int64} = copy(all_labels_) 
            best_params = Params()

            for O_ind in 1:7
                O::Vector{Float64} = _operator(O_ind, node_k_features, node_l_features)
                thresholds::Vector{Float64} = quantile(O, [i / (n_thresholds + 1)　for i in 1:n_thresholds])

                for thresh::Float64 in thresholds
                    right_mask = O .> thresh
                    left_mask = .!right_mask
                    right_indices::SubArray{Int64} = @view node_indices[right_mask]
                    left_indices::SubArray{Int64} = @view node_indices[left_mask]
                    
                    if min(length(right_indices), length(left_indices)) < min_samples
                        nothing
                    else
                        for order::Vector{Int64} in leaf_order
                            all_labels_[left_indices] .= order[1]
                            all_labels_[right_indices] .= order[2]
                            returns::Vector{Float64} = targets .* all_labels_
                            r_mean::Float64 = mean(returns)
                            score::Float64 = r_mean / stdm(returns, r_mean)

                            if score > best_score
                                best_score = score
                                best_labels[node_indices] .= all_labels_[node_indices]
                                best_params = Params(
                                    k=k, l=l, O_ind=O_ind, 
                                    thresh=thresh, order=order,
                                    left_indices=left_indices,
                                    right_indices=right_indices
                                )
                            end
                        end
                    end
                end
            end
            
            if best_score > current_best_score
                current_best_score = best_score
                all_labels = best_labels
                node.label = nothing
                node.k = best_params.k
                node.l = best_params.l
                node.O_ind = best_params.O_ind
                node.thresh = best_params.thresh
                node.left = Node(label=best_params.order[1], depth=node.depth+1, indices=best_params.left_indices)
                node.right = Node(label=best_params.order[2], depth=node.depth+1, indices=best_params.right_indices)
                
                if node.depth + 1 < max_depth
                    append!(stack, [node.left, node.right])
                end
                
                break
            end
        end    
        popat!(stack, node_ind)
    end
    _clear(root)
    return root
end

function _clear(node::Union{Node, Nothing})
    if node === nothing
        return
    else 
        node.indices = nothing
        _clear(node.left)
        _clear(node.right)
    end
end

function predict_labels(;
        node::Node, features::SubArray{Float64}
)
    n_samples = size(features)[1]
    labels = zeros(Int64, n_samples)
    node.indices = collect(1:n_samples)
    _predictor(node, features, labels)
    _clear(node)
    return labels
end

function _predictor(node::Node, features::SubArray{Float64}, labels::Vector{Int64})
    if node === nothing
        return
    end
    
    node_indices = node.indices
    
    if node.label !== nothing
        labels[node_indices] .= node.label
        return
    end

    node_k_features::SubArray{Float64} = @view features[node_indices, node.k] 
    node_l_features::SubArray{Float64} = @view features[node_indices, node.l] 

    O = _operator(node.O_ind, node_k_features, node_l_features)

    right_mask = O .> node.thresh
    left_mask = O .<= node.thresh

    node.right.indices = @view node_indices[right_mask]
    node.left.indices = @view node_indices[left_mask]

    _predictor(node.right, features, labels)
    _predictor(node.left, features, labels)
end

function _operator(O_ind::Int64, x::SubArray{Float64}, y::SubArray{Float64})
    O_list::Vector{Function} = [
            (x::SubArray{Float64}, y::SubArray{Float64}) -> x .+ y,
            (x::SubArray{Float64}, y::SubArray{Float64}) -> x .- y,
            (x::SubArray{Float64}, y::SubArray{Float64}) -> x .* y,
            (x::SubArray{Float64}, y::SubArray{Float64}) -> max.(x, y),
            (x::SubArray{Float64}, y::SubArray{Float64}) -> min.(x, y),
            (x::SubArray{Float64}, y::SubArray{Float64}) -> x,
            (x::SubArray{Float64}, y::SubArray{Float64}) -> y
        ]
    return O_list[O_ind](x, y)
end

end