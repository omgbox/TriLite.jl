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
    in_features, out_features = size(W)
    _, seq_len = size(x)
    fill!(out, 0.0f0)

    for col in 1:seq_len
        for row in 1:out_features
            acc = Int32(0)

            for k_base in 1:32:in_features
                k_end = min(k_base + 31, in_features)
                chunk_len = k_end - k_base + 1

                if chunk_len == 32
                    w_vec = _load_int8_vec_T(W, k_base, row)
                    x_enc = _load_enc_vec(x, k_base, col)
                    result = Intrinsics_x86_64.multiply_add_pairs_256(x_enc, w_vec)
                    correction = _pair_sum_w(w_vec)
                    for i in 1:16
                        acc += Int32(result[i] - correction[i])
                    end
                else
                    for k in k_base:k_end
                        acc += Int32(W[k, row]) * Int32(x[k, col])
                    end
                end
            end

            out[row, col] = Float32(acc) * scale
        end
    end
    return out
end

function matmul_mad_vec!(out::AbstractVector{Float32},
                         W::AbstractMatrix{Int8},
                         x::AbstractVector{Float32},
                         scale::Float32)
    in_features, out_features = size(W)
    fill!(out, 0.0f0)

    for row in 1:out_features
        acc = Int32(0)

        for k_base in 1:32:in_features
            k_end = min(k_base + 31, in_features)
            chunk_len = k_end - k_base + 1

            if chunk_len == 32
                w_vec = _load_int8_vec_T(W, k_base, row)
                x_enc = _load_enc_vec_vec(x, k_base)
                result = Intrinsics_x86_64.multiply_add_pairs_256(x_enc, w_vec)
                correction = _pair_sum_w(w_vec)
                for i in 1:16
                    acc += Int32(result[i] - correction[i])
                end
            else
                for k in k_base:k_end
                    acc += Int32(W[k, row]) * Int32(x[k])
                end
            end
        end

        out[row] = Float32(acc) * scale
    end
    return out
end

@inline function _load_int8_vec_T(W::AbstractMatrix{Int8}, k_base::Int, row::Int)
    vals = ntuple(i -> W[k_base + i - 1, row], Val(32))
    return SIMD.Vec(vals)
end

@inline function _load_enc_vec(x::AbstractMatrix{Float32}, k_base::Int, col::Int)
    vals = ntuple(i -> UInt8(x[k_base + i - 1, col] + 1), Val(32))
    return SIMD.Vec(vals)
end

@inline function _load_enc_vec_vec(x::AbstractVector{Float32}, k_base::Int)
    vals = ntuple(i -> UInt8(x[k_base + i - 1] + 1), Val(32))
    return SIMD.Vec(vals)
end

@inline function _pair_sum_w(w_vec::SIMD.Vec{32, Int8})
    SIMD.Vec(ntuple(Val(16)) do i
        Int16(w_vec[2i-1]) + Int16(w_vec[2i])
    end)
end
