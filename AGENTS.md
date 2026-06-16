# BitNet.jl — Agent Instructions

Pure-Julia CPU inference engine for BitNet b1.58 ternary LLMs.

## Quick Start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
```

Every command below assumes CWD is the repo root and `--project=.` is passed unless noted.

## Architecture

**Module entrypoint:** `src/BitNet.jl` — top-level module that `include()`s all subfiles into lexical scope. These are NOT Julia submodules (no `using .Foo` needed from outside), except the intrinsics modules which ARE proper submodules (`Intrinsics_x86_64`, `Intrinsics_aarch64`, `Intrinsics_Fallback`).

**CAUTION:** If you `include()` individual source files outside the module (as test files do), you must also `using .TheirSubmoduleName` for intrinsics/kernel utils. See test files for the exact pattern — mixing this up gives `UndefVarError`.

**Platform dispatch** is static, at compile time via `Sys.ARCH`:
- `src/BitNet.jl:9-15` sets `ARCH` constant
- `src/BitNet.jl:22-31` includes the correct intrinsics module
- `src/types.jl:140-146` selects default kernel trait

**Kernel dispatch** uses trait types (`RefKernel`, `MADKernel`, `LUTKernel`). The model now calls `matmul!(DEFAULT_KERNEL, ...)` / `matmul_vec!(DEFAULT_KERNEL, ...)` instead of hardcoded `matmul_ref!`. See `src/kernels/dispatch.jl`. Default: `RefKernel()` on all platforms (MAD/LUT need activation quantization).

**GGUF I2_S support:** Ternary weights packed in 2-bit format (GGML dtype 36). See `src/gguf/` for loaders, `src/gguf/repack.jl` for the unpack → Int8[-1,0,1] path.

## Dependencies

- Julia 1.12+ (`Project.toml` compat). Older versions will fail to resolve.
- `Mmap`, `SIMD` — actually used in source.
- `LoopVectorization` — listed in `Project.toml` but **nowhere used** in src/ or test/. Do not write code depending on it without adding `using LoopVectorization` first.

## Test Quirks

**IMPORTANT:** kernel source files (`src/kernels/*.jl`, `src/gguf/repack.jl`) are NOT modules — they're plain `include()` files whose names go directly into `Main`. The intrinsics modules (`Intrinsics_x86_64`, `Intrinsics_aarch64`, `Intrinsics_Fallback`, `FeatureDetect`) ARE proper Julia submodules with `export` statements; must `using .ModuleName` them.

Test loading patterns:

| File | Loads via | Risk |
|------|-----------|------|
| `test_intrinsics.jl` | `include()` + `using .FeatureDetect`, `using .Intrinsics_*` | Won't find `BitNet`-exported symbols |
| `test_kernels.jl` | `include()` kernel files + `using .Intrinsics_*` for platform dispatch | Uses qualified intrinsics names |
| `test_model.jl` | `using BitNet` (proper) | Won't have individual submodule symbols |

All must be run from repo root with `--project=.`:
```bash
julia --project=. test/runtests.jl          # all three
julia --project=. -e 'include("test/test_kernels.jl")'
julia --project=. -e 'include("test/test_model.jl")'
```

**Without `--project=.`**: `using BitNet` in `test_model.jl` will throw `ArgumentError: Package BitNet not found`.

## Root-Level Debug Scripts

`check_*.jl`, `debug_test.jl`, `probe_tensors.jl`, `test_*.jl` — ad-hoc workspace scripts, **not exported** by the package. They each `using BitNet` or `Pkg.activate(".")` internally.

`run_tests.jl` and `run_kernels.jl` — convenience wrappers that call `Pkg.activate(".")` then `include()` a test file. **Must run from repo root** (relative `include()` path).

## Loading the Package

Standard usage (from any CWD if env activated):
```julia
using BitNet
model = load_model("path/to/model.gguf")
```

The `examples/example.jl` uses `push!(LOAD_PATH, ...)` instead of `using Pkg; Pkg.activate(".")`. This only works if Julia's depot knows about BitNet.

## Known Issues (Confirmed by Running)

### load_model is slow (~460s, 100 GiB alloc)
On the small committed model (`models/ggml-model-i2_s.gguf`), `load_model()` takes ~460s and allocates ~100 GiB. The O(n²) tensor name lookup has been fixed with a `tensor_map` Dict (`src/gguf/loader.jl`), but per-layer loading still allocates heavily for each weight tensor. Progress is printed per layer. Not a hang — just pathologically slow.

### Cross-platform kernel bug
`matmul_mad.jl` and `matmul_lut.jl` hardcode `Intrinsics_x86_64.multiply_add_pairs_256` / `Intrinsics_x86_64.pshufb_128`. These will throw `UndefVarError` if called on aarch64 or fallback platforms, despite trait dispatch selecting the correct kernel. Fix: make kernel files generic or gate the hardcoded references.

## Fixed Bugs (Applied)

| Bug | Fix Location | Symptom |
|-----|-------------|---------|
| `"""..."""` docstring before `using Test` | `test/runtests.jl:1`, `test/test_intrinsics.jl:1` | Julia 1.12 `ParseError`: remove docstrings |
| `SIMD.Vec{16,UInt8}(0,1,2,...)` varargs | `test_intrinsics.jl`, `src/intrinsics/fallback.jl`, `src/kernels/matmul_lut.jl` | SIMD.jl 3.x only accepts tuple: wrap in `((...))` |
| No `export` in intrinsics modules | `src/intrinsics/{x86_64,aarch64,fallback}.jl` | `UndefVarError` after `using .Intrinsics_*` |
| x86_64 functions named without underscore prefix | `src/intrinsics/x86_64.jl` | Tests call `_split_256` etc — added aliases |
| Test used `using .KernelUtils` (not a module) | `test/test_kernels.jl:20` | `ArgumentError`: remove non-existent module imports |
| `repack_for_mad!` not found in test | `test/test_kernels.jl:39` | Function is in `src/gguf/repack.jl`, not `utils.jl` |
| `1e-6f0` parse error | `test/test_kernels.jl` | Julia parses as `1e-6 * f0` → use `Float32(1e-6)` |
| matmul_ref expected value -4 was wrong | `test/test_kernels.jl` | Correct answer is -5 per weight-vector dot product |
| MAD kernel skipped output rows | `src/kernels/matmul_mad.jl:23` | Loop used `row_base in 1:32:out_features` → only row 1 written |
| LUT table BoundsError | `src/kernels/matmul_lut.jl:45` | table was 16 elements but accessed at index 27 (27 patterns) |
| `tok_embeddings[:, token]` wrong dimension | `src/transformer/model.jl:63` | Embedding matrix is `(vocab_size, hidden_dim)`; needs `[token, :]` |
| Kernel dispatch never wired up | `src/kernels/dispatch.jl` (new), `attention.jl`, `ffn.jl` | Model hardcoded `matmul_ref!`; now dispatches via `DEFAULT_KERNEL` trait |
| Tensor lookup O(n²) linear scan | `src/gguf/loader.jl:81` | Added `tensor_map::Dict` built once at load time |
| Tokenizer not loaded from GGUF | `src/io.jl:183` | `_load_tokenizer` now reads 128k vocab + 280k merges from GGUF metadata |
| No load progress indicator | `src/io.jl:38-59` | Added per-layer timing and progress % during `load_model` |

## Important Constraints

- `models/ggml-model-i2_s.gguf` — the test model committed in-repo (small I2_S format). Loads via `BitNet.load_model("models/ggml-model-i2_s.gguf")`.
- No CI, no formatter/linter, no pre-commit hooks.
- **From parent workspace `../AGENTS.md`:** Never add `@threads` to computation loops — LLVM JIT crashes with `EXCEPTION_ACCESS_VIOLATION` on Windows. Use `@simd` only.
