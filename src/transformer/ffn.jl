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

    sum_abs = sum(abs, h_normed)
    act_scale = sum_abs / length(h_normed) + Float32(1e-6)
    h_q = Int8.(clamp.(round.(h_normed ./ act_scale), -1, 1))
    h_q_f32 = Float32.(h_q)
    cs_gate = w_gate.scale * act_scale
    cs_up = w_up.scale * act_scale

    _, out_gate = w_gate.weight_shape
    _, out_up = w_up.weight_shape
    gate = zeros(Float32, out_gate, seq_len)
    up = zeros(Float32, out_up, seq_len)
    matmul!(DEFAULT_KERNEL, gate, _unpack_ternary_ffn(w_gate), h_q_f32, cs_gate)
    matmul!(DEFAULT_KERNEL, up, _unpack_ternary_ffn(w_up), h_q_f32, cs_up)

    hidden = similar(gate)
    for i in eachindex(gate)
        hidden[i] = relu_squared(gate[i]) * up[i]
    end

    hdn_sum = sum(abs, hidden)
    hdn_scale = hdn_sum / length(hidden) + Float32(1e-6)
    hidden_q = Int8.(clamp.(round.(hidden ./ hdn_scale), -1, 1))
    hidden_q_f32 = Float32.(hidden_q)
    matmul!(DEFAULT_KERNEL, out, _unpack_ternary_ffn(w_down), hidden_q_f32, w_down.scale * hdn_scale)
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

    sum_abs = sum(abs, h_normed)
    act_scale = sum_abs / length(h_normed) + Float32(1e-6)
    h_q = Int8.(clamp.(round.(h_normed ./ act_scale), -1, 1))
    h_q_f32 = Float32.(h_q)
    cs = w_gate.scale * act_scale

    _, out_gate = w_gate.weight_shape
    _, out_up = w_up.weight_shape
    gate = zeros(Float32, out_gate)
    up = zeros(Float32, out_up)
    matmul_vec!(DEFAULT_KERNEL, gate, _unpack_ternary_ffn(w_gate), h_q_f32, cs)
    matmul_vec!(DEFAULT_KERNEL, up, _unpack_ternary_ffn(w_up), h_q_f32, cs)

    hidden = similar(gate)
    for i in eachindex(gate)
        hidden[i] = relu_squared(gate[i]) * up[i]
    end

    hdn_abs = sum(abs, hidden)
    hdn_scale = hdn_abs / length(hidden) + Float32(1e-6)
    hidden_q = Int8.(clamp.(round.(hidden ./ hdn_scale), -1, 1))
    hidden_q_f32 = Float32.(hidden_q)
    matmul_vec!(DEFAULT_KERNEL, out, _unpack_ternary_ffn(w_down), hidden_q_f32, w_down.scale * hdn_scale)
    return out
end

@inline function _unpack_ternary_ffn(bl::BitLinear)
    return reshape(Int8.(bl.packed_weights), bl.weight_shape...)
end
