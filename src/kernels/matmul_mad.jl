"""
MAD (Multiply-Add) kernel for BitNet ternary matmul.
Uses multiply-add pairs for 2× throughput over scalar.
Best for: Prefill phase (compute-bound).
"""

const MAD_SIMD_WIDTH = 32

"""
    matmul_mad!(out, W, x, scale)

MAD kernel: multiply-add using SIMD pairs.
"""
function matmul_mad!(out::AbstractMatrix{Float32},
                     W::AbstractMatrix{Int8},
                     x::AbstractMatrix{Float32},
                     scale::Float32)
    out_features, in_features = size(W)
    _, seq_len = size(x)
    fill!(out, 0.0f0)

    for col in 1:seq_len
        for row in 1:out_features
            acc = Int32(0)

            for k_base in 1:32:in_features
                k_end = min(k_base + 31, in_features)
                chunk_len = k_end - k_base + 1

                if chunk_len == 32
                    w_vec = _load_int8_vec(W, row, k_base)
                    x_vec = _load_uint8_vec(x, k_base, col)
                    result = Intrinsics_x86_64.multiply_add_pairs_256(x_vec, w_vec)
                    for i in 1:16
                        acc += Int32(result[i])
                    end
                else
                    for k in k_base:k_end
                        acc += Int32(W[row, k]) * Int32(x[k, col])
                    end
                end
            end

            out[row, col] = Float32(acc) * scale
        end
    end
    return out
end

"""
    matmul_mad_vec!(out, W, x, scale)

Vector output version for single token decode.
"""
function matmul_mad_vec!(out::AbstractVector{Float32},
                         W::AbstractMatrix{Int8},
                         x::AbstractVector{Float32},
                         scale::Float32)
    out_features, in_features = size(W)
    fill!(out, 0.0f0)

    for row in 1:out_features
        acc = Int32(0)

        for k_base in 1:32:in_features
            k_end = min(k_base + 31, in_features)
            chunk_len = k_end - k_base + 1

            if chunk_len == 32
                w_vec = _load_int8_vec(W, row, k_base)
                x_vec = _load_uint8_vec_vec(x, k_base)
                result = Intrinsics_x86_64.multiply_add_pairs_256(x_vec, w_vec)
                for i in 1:16
                    acc += Int32(result[i])
                end
            else
                for k in k_base:k_end
                    acc += Int32(W[row, k]) * Int32(x[k])
                end
            end
        end

        out[row] = Float32(acc) * scale
    end
    return out
end

# ─── Helpers ──────────────────────────────────────────────────────────

@inline function _load_int8_vec(W::AbstractMatrix{Int8}, row::Int, k_base::Int)
    vals = ntuple(i -> W[row, k_base + i - 1], Val(32))
    return SIMD.Vec(vals)
end

@inline function _load_uint8_vec(x::AbstractMatrix{Float32}, k_base::Int, col::Int)
    vals = ntuple(i -> UInt8(abs(x[k_base + i - 1, col])), Val(32))
    return SIMD.Vec(vals)
end

@inline function _load_uint8_vec_vec(x::AbstractVector{Float32}, k_base::Int)
    vals = ntuple(i -> UInt8(abs(x[k_base + i - 1])), Val(32))
    return SIMD.Vec(vals)
end
