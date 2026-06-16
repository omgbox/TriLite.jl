"""
Feed-Forward Network with ReLU² activation for BitNet models.
"""

"""
    ffn!(out, w_gate, w_up, w_down, ffn_norm_weight, h, norm_eps)

FFN for prefill phase.
"""
function ffn!(out::AbstractMatrix{Float32},
              w_gate::BitLinear, w_up::BitLinear, w_down::BitLinear,
              ffn_norm_weight::Vector{Float32},
              h::AbstractMatrix{Float32},
              norm_eps::Float32)
    hidden_dim, seq_len = size(h)

    h_normed = similar(h)
    for col in 1:seq_len
        rmsnorm!(view(h_normed, :, col), view(h, :, col), ffn_norm_weight, norm_eps)
    end

    gate = zeros(Float32, w_gate.weight_shape[1], seq_len)
    up = zeros(Float32, w_up.weight_shape[1], seq_len)
    matmul!(DEFAULT_KERNEL, gate, _unpack_ternary_ffn(w_gate), h_normed, w_gate.scale)
    matmul!(DEFAULT_KERNEL, up, _unpack_ternary_ffn(w_up), h_normed, w_up.scale)

    hidden = similar(gate)
    for i in eachindex(gate)
        hidden[i] = relu_squared(gate[i]) * up[i]
    end

    matmul!(DEFAULT_KERNEL, out, _unpack_ternary_ffn(w_down), hidden, w_down.scale)
    return out
end

"""
    ffn_decode!(out, w_gate, w_up, w_down, ffn_norm_weight, h, norm_eps)

FFN for decode phase (single token).
"""
function ffn_decode!(out::AbstractVector{Float32},
                     w_gate::BitLinear, w_up::BitLinear, w_down::BitLinear,
                     ffn_norm_weight::Vector{Float32},
                     h::AbstractVector{Float32},
                     norm_eps::Float32)
    h_normed = similar(h)
    rmsnorm!(h_normed, h, ffn_norm_weight, norm_eps)

    gate = zeros(Float32, w_gate.weight_shape[1])
    up = zeros(Float32, w_up.weight_shape[1])
    matmul_vec!(DEFAULT_KERNEL, gate, _unpack_ternary_ffn(w_gate), h_normed, w_gate.scale)
    matmul_vec!(DEFAULT_KERNEL, up, _unpack_ternary_ffn(w_up), h_normed, w_up.scale)

    hidden = similar(gate)
    for i in eachindex(gate)
        hidden[i] = relu_squared(gate[i]) * up[i]
    end

    matmul_vec!(DEFAULT_KERNEL, out, _unpack_ternary_ffn(w_down), hidden, w_down.scale)
    return out
end

@inline function _unpack_ternary_ffn(bl::BitLinear)
    return reshape(Int8.(bl.packed_weights), bl.weight_shape...)
end
