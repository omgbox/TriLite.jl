"""
Core data types for BitNet inference.
"""

# ─── Model Configuration ──────────────────────────────────────────────
"""
    BitNetConfig

Configuration for a BitNet b1.58 model, parsed from GGUF metadata.
"""
Base.@kwdef struct BitNetConfig
    hidden_dim::Int          # Model hidden dimension (e.g., 2560)
    num_heads::Int           # Number of attention heads (e.g., 20)
    num_kv_heads::Int        # Number of KV heads for GQA (e.g., 5)
    head_dim::Int            # Per-head dimension (hidden_dim / num_heads)
    num_layers::Int          # Number of transformer layers (e.g., 30)
    ffn_dim::Int             # Feed-forward network dimension (e.g., 6912)
    vocab_size::Int          # Vocabulary size (e.g., 128256)
    max_seq_len::Int         # Maximum sequence length (e.g., 4096)
    norm_eps::Float32        # RMSNorm epsilon
    rotary_dim::Int          # RoPE dimension
    rope_base::Float32       # RoPE frequency base (e.g., 500000.0)
end

# ─── Quantized Linear Layer ──────────────────────────────────────────
"""
    BitLinear

A linear layer with ternary weights {-1, 0, +1} and int8 activations.
"""
struct BitLinear
    packed_weights::Vector{Int8}  # Ternary weights {-1, 0, +1}
    weight_shape::Tuple{Int,Int}   # (out_features, in_features)
    scale::Float32                 # Per-tensor scale factor
    bias::Nothing                  # BitNet removes all biases
end

# ─── KV Cache (defined before BitNetModel) ────────────────────────────
"""
    KVCache

Pre-allocated key-value cache for autoregressive generation.
"""
mutable struct KVCache
    keys::Array{Float32, 3}   # (num_layers, max_seq_len, hidden_dim)
    values::Array{Float32, 3} # (num_layers, max_seq_len, hidden_dim)
    current_len::Int
    max_len::Int
end

function KVCache(num_layers::Int, max_len::Int, hidden_dim::Int)
    KVCache(
        zeros(Float32, num_layers, max_len, hidden_dim),
        zeros(Float32, num_layers, max_len, hidden_dim),
        0,
        max_len
    )
end

# ─── Tokenizer (defined before BitNetModel) ──────────────────────────
"""
    BitTokenizer

Minimal BPE tokenizer interface.
"""
mutable struct BitTokenizer
    vocab::Dict{String,Int}
    vocab_r::Dict{Int,String}
    merges::Vector{Tuple{Int,Int}}
    merge_map::Dict{Tuple{Int,Int},Int}  # (left_id, right_id) -> merged_id
    merge_rank::Dict{Tuple{Int,Int},Int} # (left_id, right_id) -> rank
    special_tokens::Dict{String,Int}
    bos_token::Int
    eos_token::Int
end

function BitTokenizer(vocab::Dict{String,Int}, merges::Vector{Tuple{Int,Int}},
                      special_tokens::Dict{String,Int})
    vocab_r = Dict(v => k for (k, v) in vocab)
    bos = get(special_tokens, "<|begin_of_text|>", 1)
    eos = get(special_tokens, "<|end_of_text|>", 2)

    merge_map = Dict{Tuple{Int,Int},Int}()
    merge_rank = Dict{Tuple{Int,Int},Int}()
    for (i, (id0, id1)) in enumerate(merges)
        pair = (id0, id1)
        merged_str = vocab_r[id0] * vocab_r[id1]
        merged_id = get(vocab, merged_str, length(vocab))
        merge_map[pair] = merged_id
        merge_rank[pair] = i
    end

    return BitTokenizer(vocab, vocab_r, merges, merge_map, merge_rank,
                        special_tokens, bos, eos)
end

# ─── Transformer Layer ────────────────────────────────────────────────
"""
    TransformerLayer

A single transformer block: attention + FFN with ternary weights.
"""
struct TransformerLayer
    # Attention
    wq::BitLinear          # Query projection
    wk::BitLinear          # Key projection
    wv::BitLinear          # Value projection
    wo::BitLinear          # Output projection

    # FFN
    w_gate::BitLinear      # Gate projection (SwiGLU-style)
    w_up::BitLinear        # Up projection
    w_down::BitLinear      # Down projection

    # Norms
    attention_norm_weight::Vector{Float32}   # SubLN scale
    ffn_norm_weight::Vector{Float32}         # SubLN scale
end

# ─── Full Model ───────────────────────────────────────────────────────
"""
    BitNetModel

Complete BitNet b1.58 model ready for inference.
"""
mutable struct BitNetModel
    config::BitNetConfig
    layers::Vector{TransformerLayer}

    # Non-ternary layers (stored in bf16/int8)
    tok_embeddings::Matrix{Float16}    # vocab_size × hidden_dim
    output_norm::Vector{Float32}       # Final RMSNorm scale
    lm_head::Matrix{Float16}           # Output projection (vocab_size × hidden_dim)

    # KV cache
    kv_cache::Union{Nothing, KVCache}

    # Tokenizer
    tokenizer::Union{Nothing, BitTokenizer}
end

# ─── Kernel Selection ─────────────────────────────────────────────────
"""
    KernelType

Trait type for dispatching between kernel implementations.
"""
abstract type KernelType end
struct RefKernel <: KernelType end
struct MADKernel <: KernelType end
struct LUTKernel <: KernelType end

# Default kernel selection based on platform
const DEFAULT_KERNEL = @static if ARCH == :x86_64
    MADKernel()    # x86: MAD with SIMD multiply-add pairs
elseif ARCH == :aarch64
    RefKernel()    # ARM: reference kernel
else
    RefKernel()    # Fallback: scalar reference
end
