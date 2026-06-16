"""
Token sampling strategies for BitNet models.
"""

"""
    greedy_sample(logits)

Select the token with highest logit value.
"""
function greedy_sample(logits::AbstractVector{Float32})
    return argmax(logits)
end

"""
    topk_sample(logits, k)

Sample from the top-k tokens by logit value.
"""
function topk_sample(logits::AbstractVector{Float32}, k::Int)
    sorted_indices = sortperm(logits, rev=true)
    top_k_indices = sorted_indices[1:min(k, length(logits))]
    top_k_logits = logits[top_k_indices]
    probs = softmax_vec(top_k_logits)

    r = rand()
    cumsum = 0.0f0
    for (i, prob) in enumerate(probs)
        cumsum += prob
        if r <= cumsum
            return top_k_indices[i]
        end
    end
    return top_k_indices[end]
end

"""
    topp_sample(logits, p)

Sample from the smallest set of tokens whose cumulative probability >= p.
"""
function topp_sample(logits::AbstractVector{Float32}, p::Float32)
    sorted_indices = sortperm(logits, rev=true)
    sorted_logits = logits[sorted_indices]
    probs = softmax_vec(sorted_logits)

    cumsum = 0.0f0
    nucleus_size = length(sorted_indices)
    for (i, prob) in enumerate(probs)
        cumsum += prob
        if cumsum >= p
            nucleus_size = i
            break
        end
    end

    nucleus_indices = sorted_indices[1:nucleus_size]
    nucleus_probs = probs[1:nucleus_size]
    nucleus_probs ./= sum(nucleus_probs)

    r = rand()
    cumsum = 0.0f0
    for (i, prob) in enumerate(nucleus_probs)
        cumsum += prob
        if r <= cumsum
            return nucleus_indices[i]
        end
    end
    return nucleus_indices[end]
end

"""
    sample_token(logits; temperature, top_k, top_p)

Combined sampling with temperature, top-k, and top-p.
"""
function sample_token(logits::AbstractVector{Float32};
                      temperature::Float32=1.0f0,
                      top_k::Int=0,
                      top_p::Float32=1.0f0)
    logits = copy(logits)

    if temperature != 1.0f0
        logits ./= temperature
    end

    if top_k > 0
        sorted_indices = sortperm(logits, rev=true)
        for (i, idx) in enumerate(sorted_indices)
            if i > top_k
                logits[idx] = Float32(-Inf)
            end
        end
    end

    if top_p < 1.0f0
        sorted_indices = sortperm(logits, rev=true)
        sorted_logits = logits[sorted_indices]
        probs = softmax_vec(sorted_logits)

        cumsum = 0.0f0
        for (i, prob) in enumerate(probs)
            cumsum += prob
            if cumsum >= top_p
                for j in (i+1):length(sorted_indices)
                    logits[sorted_indices[j]] = Float32(-Inf)
                end
                break
            end
        end
    end

    probs = softmax_vec(logits)
    r = rand()
    cumsum = 0.0f0
    for i in eachindex(probs)
        cumsum += probs[i]
        if r <= cumsum
            return i
        end
    end
    return length(probs)
end

"""
    softmax_vec(x) -> Vector{Float32}

Numerically stable softmax.
"""
function softmax_vec(x::AbstractVector{Float32})
    max_val = maximum(x)
    exp_x = exp.(x .- max_val)
    return exp_x ./ sum(exp_x)
end
