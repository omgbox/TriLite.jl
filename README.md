# TriLite.jl

**Pure-Julia CPU inference engine for BitNet b1.58 ternary LLMs.**

TriLite lets you run BitNet b1.58 models — large language models with {-1, 0, +1} ternary weights — on any CPU, with zero C/C++ dependencies. Built entirely in Julia. Targets Llama-class architectures (GQA, RoPE, SwiGLU, RMSNorm).

---

### What is BitNet b1.58?

BitNet b1.58 is a **ternary LLM architecture** by Microsoft Research where every weight is constrained to one of three values: **-1, 0, or +1** (1.58 bits per parameter on average). This replaces the standard FP16/BF16 weights used in models like Llama or GPT.

**Why it matters:**
- **Memory**: A 2B-parameter model fits in ~400 MB (vs ~4 GB for FP16)
- **Compute**: Multiply-accumulate becomes integer addition — no floating-point multipliers
- **Energy**: ASIC estimates show 10–100× power reduction over FP16 matmul
- **Quality**: Matches full-precision perplexity at the same parameter count

Ternary quantization is not just for inference — the model is *trained from scratch* at b1.58 precision, unlike post-training quantization approaches.

---

### The Julia Implementation

Most BitNet inference engines (microsoft/BitNet, llama.cpp) are written in C++. TriLite is a **pure-Julia reimplementation** with three kernel backends:

| Kernel | Mechanism | When to use |
|--------|-----------|-------------|
| **Reference** | Scalar `Int8 × Float32` | Correctness baseline |
| **MAD** | `vpmaddubsw` SIMD pairs | Prefill (compute-bound) |
| **LUT** | `pshufb` table lookup | Decode (memory-bound) |

The engine is **dispatch-based**: a single `matmul!(DEFAULT_KERNEL, ...)` call selects the optimal kernel at compile time via Julia's trait dispatch. No runtime branching.

Other implementation details:
- GGUF v3 model loader with `tensor_map` for O(1) tensor lookup
- GPT-2 BPE tokenizer extracted directly from GGUF metadata (no separate file needed)
- RoPE, RMSNorm, ReLU² activation, group-query attention — all in pure Julia
- Platform dispatch via `Sys.ARCH` — x86_64 (AVX2) and aarch64 (NEON) intrinsics

## Features

- **Pure Julia**: Zero C/C++ FFI — compiles with the Julia JIT
- **CPU-only**: No GPU required. x86-64 (AVX2) and ARM64 (NEON)
- **Ternary weights**: Uses 2-bit GGUF format (GGML dtype 36)
- **Chat interface**: `chat(model)` for interactive generation
- **GGUF v3**: Loads standard BitNet GGUF files from HuggingFace
- **Internal tokenizer**: GPT-2 BPE extracted from GGUF metadata

## Quick Start

```julia
using TriLite

# Load model
model = load_model("path/to/bitnet-2B.gguf")

# Generate text
response = generate(model, "What is the capital of France?")
println(response)

# Interactive chat
chat(model)
```

## Installation

```julia
using Pkg
Pkg.add("TriLite")
```

Or from source:

```bash
cd TriLite.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Requirements

- Julia 1.12+ (recommended)
- x86-64 CPU with AVX2 support, or ARM64 CPU with NEON
- 4GB+ RAM for 2B model, 16GB+ for 8B model

## Model Download

Download BitNet models from HuggingFace:

```bash
# 2B model (~400MB)
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf

# 8B model (~2GB)
huggingface-cli download microsoft/BitNet-b1.58-8B-4T-gguf
```

## Architecture

```
TriLite.jl/
├── src/
│   ├── TriLite.jl          # Main module
│   ├── types.jl           # Core data types
│   ├── intrinsics/
│   │   ├── detection.jl   # CPU feature detection
│   │   ├── x86_64.jl      # AVX2 intrinsics
│   │   ├── aarch64.jl     # NEON intrinsics
│   │   └── fallback.jl    # Scalar fallback
│   ├── kernels/
│   │   ├── utils.jl       # Kernel utilities
│   │   ├── dispatch.jl    # Trait-based kernel dispatch
│   │   ├── matmul_ref.jl  # Reference kernel
│   │   ├── matmul_mad.jl  # MAD kernel
│   │   └── matmul_lut.jl  # LUT kernel
│   ├── gguf/
│   │   ├── loader.jl      # GGUF file parser
│   │   ├── repack.jl      # Weight repacking
│   │   └── tokenizer.jl   # BPE tokenizer
│   ├── transformer/
│   │   ├── attention.jl   # Multi-head attention
│   │   ├── ffn.jl         # Feed-forward network
│   │   ├── layer.jl       # Transformer layer
│   │   └── model.jl       # Full model
│   ├── sampling.jl        # Token sampling
│   └── io.jl              # Chat interface
├── test/
│   ├── runtests.jl        # Test runner
│   ├── test_intrinsics.jl # Intrinsic tests
│   ├── test_kernels.jl    # Kernel tests
│   └── test_model.jl      # Model tests
└── Project.toml
```

## Kernels

### Reference Kernel
Scalar implementation for testing and validation.

### MAD Kernel
Uses `vpmaddubsw` for 2× throughput on x86-64.
Best for: Prefill phase (compute-bound).

### LUT Kernel
Uses `pshufb` for table lookup.
Best for: Decode phase (memory-bound).

## Performance

Currently uses the reference (scalar) kernel by default — **~0.015 tok/s on a 30-layer model** (~1 min per token). The MAD and LUT SIMD kernels are implemented and tested but need activation quantization wired into the model pipeline for a ~50–100× speedup. Contributions welcome.

## Testing

```bash
cd TriLite.jl
julia --project=. test/runtests.jl
```

## License

MIT License

## References

- [BitNet b1.58](https://arxiv.org/abs/2402.17764)
- [T-MAC](https://arxiv.org/abs/2407.00088)
- [T-SAR](https://arxiv.org/abs/2511.13676)
- [bitnet.cpp](https://github.com/microsoft/BitNet)
