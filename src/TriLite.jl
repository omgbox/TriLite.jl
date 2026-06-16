"""
BitNet.jl - Pure-Julia inference engine for BitNet b1.58 ternary LLMs.
"""
module TriLite

using Mmap
using SIMD

const ARCH = @static if Sys.ARCH == :x86_64 || Sys.ARCH == :i686
    :x86_64
elseif Sys.ARCH == :aarch64 || Sys.ARCH == :arm64
    :aarch64
else
    :fallback
end

# ─── Core Types ───────────────────────────────────────────────────────
include("types.jl")

# ─── SIMD Intrinsics (platform-specific) ─────────────────────────────
include("intrinsics/detection.jl")
@static if ARCH == :x86_64
    include("intrinsics/x86_64.jl")
    using .Intrinsics_x86_64
elseif ARCH == :aarch64
    include("intrinsics/aarch64.jl")
    using .Intrinsics_aarch64
else
    include("intrinsics/fallback.jl")
    using .Intrinsics_Fallback
end

# ─── Kernel Utilities & Kernels ──────────────────────────────────────
include("kernels/utils.jl")
include("kernels/matmul_ref.jl")
include("kernels/matmul_mad.jl")
include("kernels/matmul_lut.jl")
include("kernels/dispatch.jl")

# ─── GGUF Model Loading ──────────────────────────────────────────────
include("gguf/loader.jl")
include("gguf/repack.jl")
include("gguf/tokenizer.jl")

# ─── Transformer Architecture ─────────────────────────────────────────
include("transformer/attention.jl")
include("transformer/ffn.jl")
include("transformer/layer.jl")
include("transformer/model.jl")

# ─── Sampling & Generation ───────────────────────────────────────────
include("sampling.jl")
include("io.jl")

# ─── Public API ───────────────────────────────────────────────────────
export load_model, generate, chat, bitnet_forward!, bitnet_forward_decode!
export BitNetConfig, BitNetModel, TransformerLayer, BitLinear, KVCache
export apply_rope!, apply_rope_single!, rmsnorm!, rmsnorm, softmax!, softmax_vec
export greedy_sample, topk_sample, topp_sample, sample_token
export relu_squared, absmean_quantize, matmul_ref!, matmul_ref_vec!
export load_gguf, load_tensor, load_tensor_i8, extract_config

end # module TriLite
