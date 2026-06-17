"""
LUT (Look-Up Table) kernel for BitNet ternary matmul.
Uses pshufb for 16 parallel table lookups per instruction.
Best for: Decode phase (memory-bound).
"""

const LUT_GROUP_SIZE = 3
const LUT_NUM_PATTERNS = 27

"""
    LUTTables

Pre-computed lookup tables for LUT kernel.
"""
struct LUTTables
    tables::Vector{SIMD.Vec{16,UInt8}}
    scale::Vector{Float32}
    num_rows::Int
end

"""
    build_lut_tables!(W, x_ref, scale) -> LUTTables

Pre-compute LUT tables from weight matrix.
"""
function build_lut_tables!(W::AbstractMatrix{Int8}, x_ref::AbstractVector{Float32},
                           scale::Float32)
    in_features, out_features = size(W)
    num_groups = cld(in_features, LUT_GROUP_SIZE)
    tables = Vector{SIMD.Vec{16,UInt8}}(undef, out_features)

    for row in 1:out_features
        table = zeros(UInt8, LUT_NUM_PATTERNS)
        for g in 1:min(num_groups, 16)
            base_idx = (g - 1) * LUT_GROUP_SIZE + 1
            w0 = base_idx <= in_features ? W[base_idx, row] : Int8(0)
            w1 = base_idx + 1 <= in_features ? W[base_idx + 1, row] : Int8(0)
            w2 = base_idx + 2 <= in_features ? W[base_idx + 2, row] : Int8(0)

            for pattern in 0:(LUT_NUM_PATTERNS - 1)
                x0 = _decode_ternary(pattern, 0)
                x1 = _decode_ternary(pattern, 1)
                x2 = _decode_ternary(pattern, 2)
                partial = w0 * x0 + w1 * x1 + w2 * x2
                table[pattern + 1] = UInt8(partial + 3)
            end
        end
        tables[row] = SIMD.Vec(ntuple(i -> i <= 27 ? table[i] : UInt8(0), Val(16)))
    end

    return LUTTables(tables, [scale], out_features)
end

@inline function _decode_ternary(pattern::Int, bit_pos::Int)
    bits = (pattern >> (bit_pos * 2)) & 0x03
    return bits == 0 ? Int8(-1) : (bits == 1 ? Int8(0) : Int8(1))
end

"""
    matmul_lut!(out, lut_tables, x, scale)

LUT kernel: use pshufb for fast ternary matmul.
"""
function matmul_lut!(out::AbstractMatrix{Float32},
                     lut_tables::LUTTables,
                     x::AbstractMatrix{Float32},
                     scale::Float32)
    out_features = lut_tables.num_rows
    _, seq_len = size(x)
    fill!(out, 0.0f0)

    for col in 1:seq_len
        idx_vec = _build_index_vector(x, col)

        for row in 1:out_features
            table = lut_tables.tables[row]
            partial_sums = Intrinsics_x86_64.pshufb_128(table, idx_vec)
            acc = Int32(0)
            for i in 1:16
                acc += Int32(partial_sums[i])
            end
            out[row, col] = Float32(acc - 3) * scale
        end
    end
    return out
end

"""
    matmul_lut_vec!(out, lut_tables, x, scale)

Vector output version for single token decode.
"""
function matmul_lut_vec!(out::AbstractVector{Float32},
                         lut_tables::LUTTables,
                         x::AbstractVector{Float32},
                         scale::Float32)
    out_features = lut_tables.num_rows
    fill!(out, 0.0f0)
    idx_vec = _build_index_vector_vec(x)

    for row in 1:out_features
        table = lut_tables.tables[row]
        partial_sums = Intrinsics_x86_64.pshufb_128(table, idx_vec)
        acc = Int32(0)
        for i in 1:16
            acc += Int32(partial_sums[i])
        end
        out[row] = Float32(acc - 3) * scale
    end
    return out
end

# ─── Helpers ──────────────────────────────────────────────────────────

@inline function _build_index_vector(x::AbstractMatrix{Float32}, col::Int)
    indices = ntuple(Val(16)) do g
        base = (g - 1) * LUT_GROUP_SIZE + 1
        if base + 2 <= size(x, 1)
            x0 = _ternarize(x[base, col])
            x1 = _ternarize(x[base + 1, col])
            x2 = _ternarize(x[base + 2, col])
            _encode_ternary_index(x0, x1, x2)
        else
            UInt8(0)
        end
    end
    return SIMD.Vec(indices)
end

@inline function _build_index_vector_vec(x::AbstractVector{Float32})
    indices = ntuple(Val(16)) do g
        base = (g - 1) * LUT_GROUP_SIZE + 1
        if base + 2 <= length(x)
            x0 = _ternarize(x[base])
            x1 = _ternarize(x[base + 1])
            x2 = _ternarize(x[base + 2])
            _encode_ternary_index(x0, x1, x2)
        else
            UInt8(0)
        end
    end
    return SIMD.Vec(indices)
end

@inline function _ternarize(x::Float32)
    x < 0.0f0 ? Int8(-1) : (x > 0.0f0 ? Int8(1) : Int8(0))
end

@inline function _encode_ternary_index(x0::Int8, x1::Int8, x2::Int8)
    b0 = x0 == -1 ? 0 : (x0 == 0 ? 1 : 2)
    b1 = x1 == -1 ? 0 : (x1 == 0 ? 1 : 2)
    b2 = x2 == -1 ? 0 : (x2 == 0 ? 1 : 2)
    return UInt8(b0 | (b1 << 2) | (b2 << 4))
end
