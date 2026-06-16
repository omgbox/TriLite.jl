"""
Transformer layer: Attention + FFN with SubLN residual connections.
"""

"""
    transformer_layer!(out, layer, h, cache, layer_idx, pos, norm_eps, num_heads, num_kv_heads, head_dim, rope_base)

Single transformer layer for prefill phase.
"""
function transformer_layer!(out::AbstractMatrix{Float32},
                            layer::TransformerLayer,
                            h::AbstractMatrix{Float32},
                            cache::KVCache,
                            layer_idx::Int,
                            pos::Int, norm_eps::Float32,
                            num_heads::Int, num_kv_heads::Int, head_dim::Int, rope_base::Float32)
    att_out = similar(h)
    attention!(att_out, cache, layer_idx,
               layer.wq, layer.wk, layer.wv, layer.wo,
               layer.attention_norm_weight, h, pos, norm_eps,
               num_heads, num_kv_heads, head_dim, rope_base)

    for i in eachindex(out)
        out[i] = h[i] + att_out[i]
    end

    ffn_out = similar(out)
    ffn!(ffn_out, layer.w_gate, layer.w_up, layer.w_down,
         layer.ffn_norm_weight, out, norm_eps)

    for i in eachindex(out)
        out[i] = out[i] + ffn_out[i]
    end
    return out
end

"""
    transformer_layer_decode!(out, layer, h, cache, layer_idx, pos, norm_eps, num_heads, num_kv_heads, head_dim, rope_base)

Single transformer layer for decode phase.
"""
function transformer_layer_decode!(out::AbstractVector{Float32},
                                   layer::TransformerLayer,
                                   h::AbstractVector{Float32},
                                   cache::KVCache,
                                   layer_idx::Int,
                                   pos::Int, norm_eps::Float32,
                                   num_heads::Int, num_kv_heads::Int, head_dim::Int, rope_base::Float32)
    att_out = similar(h)
    attention_decode!(att_out, cache, layer_idx,
                      layer.wq, layer.wk, layer.wv, layer.wo,
                      layer.attention_norm_weight, h, pos, norm_eps,
                      num_heads, num_kv_heads, head_dim, rope_base)

    for i in eachindex(out)
        out[i] = h[i] + att_out[i]
    end

    ffn_out = similar(out)
    ffn_decode!(ffn_out, layer.w_gate, layer.w_up, layer.w_down,
                layer.ffn_norm_weight, out, norm_eps)

    for i in eachindex(out)
        out[i] = out[i] + ffn_out[i]
    end
    return out
end
