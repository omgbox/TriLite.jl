"""
Multi-head attention with KV cache for BitNet models.
"""

function attention!(out::AbstractMatrix{Float32},
                    cache::KVCache,
                    layer_idx::Int,
                    wq::BitLinear, wk::BitLinear, wv::BitLinear, wo::BitLinear,
                    att_norm_weight::Vector{Float32},
                    h::AbstractMatrix{Float32},
                    pos::Int, norm_eps::Float32,
                    num_heads::Int, num_kv_heads::Int, head_dim::Int, rope_base::Float32)
    hidden_dim = size(h, 1)
    seq_len = size(h, 2)
    kv_dim = num_kv_heads * head_dim

    h_normed = similar(h)
    for col in 1:seq_len
        rmsnorm!(view(h_normed, :, col), view(h, :, col), att_norm_weight, norm_eps)
    end

    sum_abs = sum(abs, h_normed)
    act_scale = sum_abs / length(h_normed) + Float32(1e-6)
    h_q = Int8.(clamp.(round.(h_normed ./ act_scale), -1, 1))
    h_q_f32 = Float32.(h_q)
    cs_q = wq.scale * act_scale
    cs_k = wk.scale * act_scale
    cs_v = wv.scale * act_scale

    q = zeros(Float32, hidden_dim, seq_len)
    k = zeros(Float32, kv_dim, seq_len)
    v = zeros(Float32, kv_dim, seq_len)
    matmul!(DEFAULT_KERNEL, q, _unpack_ternary(wq), h_q_f32, cs_q)
    matmul!(DEFAULT_KERNEL, k, _unpack_ternary(wk), h_q_f32, cs_k)
    matmul!(DEFAULT_KERNEL, v, _unpack_ternary(wv), h_q_f32, cs_v)

    for t in 1:seq_len
        for h in 1:num_heads
            offset = (h - 1) * head_dim
            apply_rope_single!(view(q, offset+1:offset+head_dim, t), pos + t - 1, head_dim, rope_base)
        end
        for h in 1:num_kv_heads
            offset = (h - 1) * head_dim
            apply_rope_single!(view(k, offset+1:offset+head_dim, t), pos + t - 1, head_dim, rope_base)
        end
    end

    if cache.current_len + seq_len <= cache.max_len
        start_idx = cache.current_len + 1
        end_idx = cache.current_len + seq_len
        cache.keys[layer_idx, start_idx:end_idx, 1:kv_dim] .= k'
        cache.values[layer_idx, start_idx:end_idx, 1:kv_dim] .= v'
        cache.current_len = end_idx
    end

    att_out = _compute_attention(q, k, v, num_heads, num_kv_heads, head_dim, seq_len)

    att_sum = sum(abs, att_out)
    att_scale = att_sum / length(att_out) + Float32(1e-6)
    att_q = Int8.(clamp.(round.(att_out ./ att_scale), -1, 1))
    att_q_f32 = Float32.(att_q)
    matmul!(DEFAULT_KERNEL, out, _unpack_ternary(wo), att_q_f32, wo.scale * att_scale)
    return out
end

function attention_decode!(out::AbstractVector{Float32},
                           cache::KVCache,
                           layer_idx::Int,
                           wq::BitLinear, wk::BitLinear, wv::BitLinear, wo::BitLinear,
                           att_norm_weight::Vector{Float32},
                           h::AbstractVector{Float32},
                           pos::Int, norm_eps::Float32,
                           num_heads::Int, num_kv_heads::Int, head_dim::Int, rope_base::Float32)
    hidden_dim = length(h)
    kv_dim = num_kv_heads * head_dim

    h_normed = similar(h)
    rmsnorm!(h_normed, h, att_norm_weight, norm_eps)

    sum_abs = sum(abs, h_normed)
    act_scale = sum_abs / length(h_normed) + Float32(1e-6)
    h_q = Int8.(clamp.(round.(h_normed ./ act_scale), -1, 1))
    h_q_f32 = Float32.(h_q)
    cs = wq.scale * act_scale

    q = zeros(Float32, hidden_dim)
    k = zeros(Float32, kv_dim)
    v = zeros(Float32, kv_dim)
    matmul_vec!(DEFAULT_KERNEL, q, _unpack_ternary(wq), h_q_f32, cs)
    matmul_vec!(DEFAULT_KERNEL, k, _unpack_ternary(wk), h_q_f32, cs)
    matmul_vec!(DEFAULT_KERNEL, v, _unpack_ternary(wv), h_q_f32, cs)

    for h in 1:num_heads
        offset = (h - 1) * head_dim
        apply_rope_single!(view(q, offset+1:offset+head_dim), pos, head_dim, rope_base)
    end
    for h in 1:num_kv_heads
        offset = (h - 1) * head_dim
        apply_rope_single!(view(k, offset+1:offset+head_dim), pos, head_dim, rope_base)
    end

    if cache.current_len < cache.max_len
        cache.current_len += 1
        cache.keys[layer_idx, cache.current_len, 1:kv_dim] .= k
        cache.values[layer_idx, cache.current_len, 1:kv_dim] .= v
    end

    att_out = _compute_attention_decode(q, cache, layer_idx, num_heads, num_kv_heads, head_dim, pos)

    att_sum = sum(abs, att_out)
    att_scale = att_sum / length(att_out) + Float32(1e-6)
    att_q = Int8.(clamp.(round.(att_out ./ att_scale), -1, 1))
    att_q_f32 = Float32.(att_q)
    matmul_vec!(DEFAULT_KERNEL, out, _unpack_ternary(wo), att_q_f32, wo.scale * att_scale)
    return out
end

@inline function _unpack_ternary(bl::BitLinear)
    return reshape(Int8.(bl.packed_weights), bl.weight_shape...)
end

function _compute_attention(q::AbstractMatrix{Float32}, k::AbstractMatrix{Float32},
                            v::AbstractMatrix{Float32}, num_heads::Int,
                            num_kv_heads::Int, head_dim::Int, seq_len::Int)
    hidden_dim = num_heads * head_dim
    kv_group_size = num_heads ÷ num_kv_heads
    scale = 1.0f0 / sqrt(Float32(head_dim))

    q_r = reshape(q, head_dim, num_heads, seq_len)
    k_r = reshape(k, head_dim, num_kv_heads, seq_len)
    v_r = reshape(v, head_dim, num_kv_heads, seq_len)

    att_scores = zeros(Float32, num_heads, seq_len, seq_len)
    for h in 1:num_heads, i in 1:seq_len, j in 1:seq_len
        kv_h = cld(h, kv_group_size)
        s = 0.0f0
        for d in 1:head_dim
            s += q_r[d, h, i] * k_r[d, kv_h, j]
        end
        att_scores[h, i, j] = s * scale
    end

    for h in 1:num_heads, i in 1:seq_len, j in (i+1):seq_len
        att_scores[h, i, j] = Float32(-Inf)
    end

    att_weights = zeros(Float32, num_heads, seq_len, seq_len)
    for h in 1:num_heads, i in 1:seq_len
        softmax!(view(att_weights, h, i, :), view(att_scores, h, i, :))
    end

    out = zeros(Float32, hidden_dim, seq_len)
    out_r = reshape(out, head_dim, num_heads, seq_len)
    for h in 1:num_heads, i in 1:seq_len, d in 1:head_dim
        kv_h = cld(h, kv_group_size)
        acc = 0.0f0
        for j in 1:seq_len
            acc += att_weights[h, i, j] * v_r[d, kv_h, j]
        end
        out_r[d, h, i] = acc
    end
    return out
end

function _compute_attention_decode(q::AbstractVector{Float32}, cache::KVCache,
                                   layer_idx::Int, num_heads::Int, num_kv_heads::Int,
                                   head_dim::Int, pos::Int)
    hidden_dim = num_heads * head_dim
    seq_len = cache.current_len
    kv_group_size = num_heads ÷ num_kv_heads
    scale = 1.0f0 / sqrt(Float32(head_dim))

    q_r = reshape(q, head_dim, num_heads)
    att_scores = zeros(Float32, num_heads, seq_len)

    for h in 1:num_heads, j in 1:seq_len
        kv_h = cld(h, kv_group_size)
        s = 0.0f0
        for d in 1:head_dim
            s += q_r[d, h] * cache.keys[layer_idx, j, (kv_h-1)*head_dim+d]
        end
        att_scores[h, j] = s * scale
    end

    att_weights = zeros(Float32, num_heads, seq_len)
    for h in 1:num_heads
        softmax!(view(att_weights, h, :), view(att_scores, h, :))
    end

    out = zeros(Float32, hidden_dim)
    out_r = reshape(out, head_dim, num_heads)
    for h in 1:num_heads, d in 1:head_dim
        kv_h = cld(h, kv_group_size)
        acc = 0.0f0
        for j in 1:seq_len
            acc += att_weights[h, j] * cache.values[layer_idx, j, (kv_h-1)*head_dim+d]
        end
        out_r[d, h] = acc
    end
    return out
end
