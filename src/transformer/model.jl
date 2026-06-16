"""
Full BitNet transformer model forward pass.
"""

"""
    bitnet_forward!(logits, model, tokens, pos)

Full forward pass for prefill phase.
"""
function bitnet_forward!(logits::AbstractMatrix{Float32},
                         model::BitNetModel,
                         tokens::AbstractVector{Int},
                         pos::Int)
    config = model.config
    seq_len = length(tokens)

    # Token embeddings
    h = zeros(Float32, config.hidden_dim, seq_len)
    for (i, token) in enumerate(tokens)
        h[:, i] .= Float32.(model.tok_embeddings[token, :])
    end

    model.kv_cache.current_len = 0

    # Transformer layers
    h_new = similar(h)
    for (layer_idx, layer) in enumerate(model.layers)
        transformer_layer!(h_new, layer, h, model.kv_cache,
                           layer_idx, pos, config.norm_eps,
                           config.num_heads, config.num_kv_heads,
                           config.head_dim, config.rope_base)
        h .= h_new
    end

    # Final RMSNorm
    h_final = similar(h)
    for col in 1:seq_len
        rmsnorm!(view(h_final, :, col), view(h, :, col),
                 model.output_norm, config.norm_eps)
    end

    # LM head (float matmul, not ternary)
    lm_head_f = Float32.(model.lm_head)
    lm_out = zeros(Float32, size(lm_head_f, 1), seq_len)
    for col in 1:seq_len
        matmul_ref_float_vec!(view(lm_out, :, col), lm_head_f, view(h_final, :, col), 1.0f0)
    end
    logits .= lm_out
    return logits
end

"""
    bitnet_forward_decode!(logits, model, token, pos)

Forward pass for single token generation (decode phase).
"""
function bitnet_forward_decode!(logits::AbstractVector{Float32},
                                model::BitNetModel,
                                token::Int,
                                pos::Int)
    config = model.config

    h = Float32.(model.tok_embeddings[token, :])

    h_new = similar(h)
    for (layer_idx, layer) in enumerate(model.layers)
        transformer_layer_decode!(h_new, layer, h, model.kv_cache,
                                  layer_idx, pos, config.norm_eps,
                                  config.num_heads, config.num_kv_heads,
                                  config.head_dim, config.rope_base)
        h .= h_new
    end

    h_final = similar(h)
    rmsnorm!(h_final, h, model.output_norm, config.norm_eps)

    matmul_ref_float_vec!(logits, Float32.(model.lm_head), h_final, 1.0f0)
    return logits
end

# Float matmul for non-ternary layers (embeddings, lm_head)
function matmul_ref_float_vec!(out::AbstractVector{Float32},
                               W::AbstractMatrix{Float32},
                               x::AbstractVector{Float32},
                               scale::Float32)
    out_features, in_features = size(W)
    fill!(out, 0.0f0)
    for row in 1:out_features
        acc = 0.0f0
        for k in 1:in_features
            acc += W[row, k] * x[k]
        end
        out[row] = acc * scale
    end
    return out
end
