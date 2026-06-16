# TriLite.jl

Pure-Julia CPU inference engine for BitNet b1.58 ternary LLMs.

## Features

- **Pure Julia**: No C/C++ FFI required
- **CPU-only**: Optimized for x86-64 (AVX2) and ARM64 (NEON)
- **Fast inference**: Targets 20+ tok/s on 2B models
- **Memory efficient**: ~1GB RAM for 2B model
- **Multiple kernels**: Reference, MAD, and LUT implementations

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

Target performance on modern CPUs:

| Model | Size | 2B Model | 8B Model |
|-------|------|----------|----------|
| Prefill | - | 20+ tok/s | 5+ tok/s |
| Decode | - | 20+ tok/s | 5+ tok/s |
| Memory | - | ~1GB | ~4GB |

## Testing

```bash
cd TriLite.jl
julia --project test/runtests.jl
```

## License

MIT License

## References

- [BitNet b1.58](https://arxiv.org/abs/2402.17764)
- [T-MAC](https://arxiv.org/abs/2407.00088)
- [T-SAR](https://arxiv.org/abs/2511.13676)
- [bitnet.cpp](https://github.com/microsoft/BitNet)
