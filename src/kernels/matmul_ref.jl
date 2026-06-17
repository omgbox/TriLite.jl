"""
Reference kernel: scalar ternary matmul for testing and validation.
"""

"""
    matmul_ref!(out, W, x, scale)

Ternary matmul: out = scale * W * x where W ∈ {-1, 0, +1}
"""
function matmul_ref!(out::AbstractMatrix{Float32},
                     W::AbstractMatrix{Int8},
                     x::AbstractMatrix{Float32},
                     scale::Float32)
    in_features, out_features = size(W)
    _, seq_len = size(x)
    fill!(out, 0.0f0)
    for col in 1:seq_len
        for row in 1:out_features
            acc = 0.0f0
            @simd for k in 1:in_features
                acc += W[k, row] * x[k, col]
            end
            out[row, col] = acc * scale
        end
    end
    return out
end

function matmul_ref(W::AbstractMatrix{Int8}, x::AbstractMatrix{Float32}, scale::Float32)
    in_features, out_features = size(W)
    _, seq_len = size(x)
    out = zeros(Float32, out_features, seq_len)
    matmul_ref!(out, W, x, scale)
    return out
end

function matmul_ref_vec!(out::AbstractVector{Float32},
                         W::AbstractMatrix{Int8},
                         x::AbstractVector{Float32},
                         scale::Float32)
    in_features, out_features = size(W)
    fill!(out, 0.0f0)
    for row in 1:out_features
        acc = 0.0f0
        @simd for k in 1:in_features
            acc += W[k, row] * x[k]
        end
        out[row] = acc * scale
    end
    return out
end
