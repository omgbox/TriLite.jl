"""
Kernel utility functions for BitNet matmul operations.
"""

# ─── Ternary Quantization ────────────────────────────────────────────

"""
    absmean_quantize!(out, w)

Ternary quantization: W̃ = RoundClip(W / (mean(|W|) + ε), -1, +1)
"""
@inline function absmean_quantize!(out::AbstractVector{Int8}, w::AbstractVector{Float32})
    epsilon = Float32(1e-6)
    sum_abs = Float32(0.0)
    for i in eachindex(w)
        sum_abs += abs(w[i])
    end
    scale = 1.0f0 / (sum_abs / length(w) + epsilon)
    for i in eachindex(w)
        scaled = w[i] * scale
        out[i] = Int8(clamp(round(scaled), -1, 1))
    end
    return out
end

function absmean_quantize(w::AbstractVector{Float32})
    out = zeros(Int8, length(w))
    absmean_quantize!(out, w)
    return out
end

# ─── Weight Repacking ─────────────────────────────────────────────────

@inline function ceil_to_multiple(n::Int, multiple::Int)
    return cld(n, multiple) * multiple
end

# ─── Activation Function ─────────────────────────────────────────────

@inline function relu_squared(x::Float32)
    return x > 0 ? x * x : 0.0f0
end

# ─── RMSNorm ──────────────────────────────────────────────────────────

@inline function rmsnorm!(out::AbstractVector{Float32}, x::AbstractVector{Float32},
                          weight::AbstractVector{Float32}, eps::Float32)
    n = length(x)
    sum_sq = 0.0f0
    for i in 1:n
        sum_sq += x[i] * x[i]
    end
    inv_rms = 1.0f0 / sqrt(sum_sq / n + eps)
    for i in 1:n
        out[i] = x[i] * inv_rms * weight[i]
    end
    return out
end

function rmsnorm(x::AbstractVector{Float32}, weight::AbstractVector{Float32}, eps::Float32)
    out = similar(x)
    rmsnorm!(out, x, weight, eps)
    return out
end

# ─── Softmax ──────────────────────────────────────────────────────────

@inline function softmax!(out::AbstractVector{Float32}, x::AbstractVector{Float32})
    max_val = maximum(x)
    sum_exp = 0.0f0
    for i in eachindex(x)
        out[i] = exp(x[i] - max_val)
        sum_exp += out[i]
    end
    inv_sum = 1.0f0 / sum_exp
    for i in eachindex(out)
        out[i] *= inv_sum
    end
    return out
end

# ─── RoPE ─────────────────────────────────────────────────────────────

@inline function apply_rope_single!(vec::AbstractVector{Float32},
                                    pos::Int, dim::Int, base::Float32=10000.0f0)
    for i in 1:2:dim
        freq = 1.0f0 / base^(Float32(i) / Float32(dim))
        theta = Float32(pos) * freq
        cos_t = cos(theta)
        sin_t = sin(theta)
        v1 = vec[i]; v2 = vec[i+1]
        vec[i]   = v1 * cos_t - v2 * sin_t
        vec[i+1] = v1 * sin_t + v2 * cos_t
    end
    return vec
end

@inline function apply_rope!(q::AbstractVector{Float32}, k::AbstractVector{Float32},
                             pos::Int, dim::Int, base::Float32=10000.0f0)
    apply_rope_single!(q, pos, dim, base)
    apply_rope_single!(k, pos, dim, base)
    return (q, k)
end
