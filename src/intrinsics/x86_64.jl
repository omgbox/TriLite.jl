"""
x86_64 SIMD intrinsics for BitNet ternary operations.

Uses SIMD.jl for portable SIMD operations and ccall(llvmcall) only
where SIMD.jl doesn't provide the needed operations.

Key operations:
- pshufb: Table lookup via permute bytes (used in LUT kernel)
- multiply-add pairs: Used in MAD kernel
"""
module Intrinsics_x86_64

using SIMD

export pshufb_128, pshufb_256,
       multiply_add_pairs_256, multiply_add_pairs_128,
       vpmaddubsw_256, vpmaddubsw_128,
       split_256, _split_256, combine_256, _combine_256,
       broadcast_256, _broadcast_256, zero_256, _zero_256, zero_256_int8, _zero_256_int8,
       load_256, store_256!, load_128, store_128!,
       hsum_256_i16, hsum_128_i16

# ─── 128-bit Operations (SSE) ─────────────────────────────────────────

"""
    pshufb_128(a::SIMD.Vec{16,UInt8}, indices::SIMD.Vec{16,UInt8}) -> SIMD.Vec{16,UInt8}

Byte-wise table lookup using 1 table (16 bytes).
Each byte in `indices` selects from `a` using low 4 bits as index.
If bit 6 of the index byte is set, result byte is 0.
"""
@inline function pshufb_128(a::SIMD.Vec{16,UInt8}, indices::SIMD.Vec{16,UInt8})
    return SIMD.Vec(ntuple(Val(16)) do i
        idx = indices[i]
        (idx & 0x80) != 0 ? UInt8(0) : a[(idx & 0x0f) + 1]
    end)
end

"""
    pshufb_256(a::SIMD.Vec{32,UInt8}, indices::SIMD.Vec{32,UInt8}) -> SIMD.Vec{32,UInt8}

AVX2 byte shuffle — 256-bit, 32 bytes in parallel.
Low 4 bits of each index byte select from 0-15 of each 128-bit lane.
"""
@inline function pshufb_256(a::SIMD.Vec{32,UInt8}, indices::SIMD.Vec{32,UInt8})
    return SIMD.Vec(ntuple(Val(32)) do i
        idx = indices[i]
        if (idx & 0x80) != 0
            UInt8(0)
        else
            lane_offset = ((i - 1) ÷ 16) * 16
            a[lane_offset + (idx & 0x0f) + 1]
        end
    end)
end

# ─── Multiply-Add Operations ──────────────────────────────────────────

"""
    multiply_add_pairs_256(a::SIMD.Vec{32,UInt8}, b::SIMD.Vec{32,Int8}) -> SIMD.Vec{16,Int16}

Multiply unsigned × signed adjacent pairs, sum each pair.
Equivalent to vpmaddubsw on x86-64.
  result[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]
"""
@inline function multiply_add_pairs_256(a::SIMD.Vec{32,UInt8}, b::SIMD.Vec{32,Int8})
    return SIMD.Vec(ntuple(Val(16)) do i
        Int16(a[2*(i-1)+1]) * Int16(b[2*(i-1)+1]) + Int16(a[2*(i-1)+2]) * Int16(b[2*(i-1)+2])
    end)
end

"""
    multiply_add_pairs_128(a::SIMD.Vec{16,UInt8}, b::SIMD.Vec{16,Int8}) -> SIMD.Vec{8,Int16}

128-bit multiply-add adjacent pairs.
"""
@inline function multiply_add_pairs_128(a::SIMD.Vec{16,UInt8}, b::SIMD.Vec{16,Int8})
    return SIMD.Vec(ntuple(Val(8)) do i
        Int16(a[2*(i-1)+1]) * Int16(b[2*(i-1)+1]) + Int16(a[2*(i-1)+2]) * Int16(b[2*(i-1)+2])
    end)
end

# ─── Helper Functions ─────────────────────────────────────────────────

"""
    split_256(v::SIMD.Vec{32,T}) -> (SIMD.Vec{16,T}, SIMD.Vec{16,T})

Split a 256-bit vector into low and high 128-bit halves.
"""
@inline function split_256(v::SIMD.Vec{32,T}) where T
    lo = SIMD.Vec(ntuple(i -> v[i], Val(16)))
    hi = SIMD.Vec(ntuple(i -> v[i + 16], Val(16)))
    return (lo, hi)
end

"""
    combine_256(lo::SIMD.Vec{16,T}, hi::SIMD.Vec{16,T}) -> SIMD.Vec{32,T}

Combine two 128-bit vectors into a 256-bit vector.
"""
@inline function combine_256(lo::SIMD.Vec{16,T}, hi::SIMD.Vec{16,T}) where T
    return SIMD.Vec(ntuple(i -> i <= 16 ? lo[i] : hi[i - 16], Val(32)))
end

"""
    broadcast_256(val::UInt8) -> SIMD.Vec{32,UInt8}

Broadcast a single byte to all 32 positions.
"""
@inline function broadcast_256(val::UInt8)
    return SIMD.Vec(ntuple(_ -> val, Val(32)))
end

"""
    broadcast_256(val::Int8) -> SIMD.Vec{32,Int8}

Broadcast a single signed byte to all 32 positions.
"""
@inline function broadcast_256(val::Int8)
    return SIMD.Vec(ntuple(_ -> val, Val(32)))
end

"""
    zero_256() -> SIMD.Vec{32,UInt8}

Return a zeroed 256-bit unsigned vector.
"""
@inline function zero_256()
    return broadcast_256(UInt8(0))
end

"""
    zero_256_int8() -> SIMD.Vec{32,Int8}

Return a zeroed 256-bit signed vector.
"""
@inline function zero_256_int8()
    return broadcast_256(Int8(0))
end

# ─── Load/Store Operations ───────────────────────────────────────────

"""
    load_256(ptr::Ptr{T}) -> SIMD.Vec{32,T}

Load 32 bytes from memory (unaligned).
"""
@inline function load_256(ptr::Ptr{T}) where T
    return unsafe_load(Ptr{SIMD.Vec{32,T}}(ptr))
end

"""
    store_256!(ptr::Ptr{T}, val::SIMD.Vec{32,T})

Store 32 bytes to memory (unaligned).
"""
@inline function store_256!(ptr::Ptr{T}, val::SIMD.Vec{32,T}) where T
    unsafe_store!(Ptr{SIMD.Vec{32,T}}(ptr), val)
end

"""
    load_128(ptr::Ptr{T}) -> SIMD.Vec{16,T}

Load 16 bytes from memory (unaligned).
"""
@inline function load_128(ptr::Ptr{T}) where T
    return unsafe_load(Ptr{SIMD.Vec{16,T}}(ptr))
end

"""
    store_128!(ptr::Ptr{T}, val::SIMD.Vec{16,T})

Store 16 bytes to memory (unaligned).
"""
@inline function store_128!(ptr::Ptr{T}, val::SIMD.Vec{16,T}) where T
    unsafe_store!(Ptr{SIMD.Vec{16,T}}(ptr), val)
end

# ─── Horizontal Sum ───────────────────────────────────────────────────

"""
    hsum_256_i16(v::SIMD.Vec{16,Int16}) -> Int32

Horizontal sum of 16 Int16 values in a 256-bit register.
"""
@inline function hsum_256_i16(v::SIMD.Vec{16,Int16})
    s = Int32(0)
    for i in 1:16
        s += Int32(v[i])
    end
    return s
end

"""
    hsum_128_i16(v::SIMD.Vec{8,Int16}) -> Int32

Horizontal sum of 8 Int16 values in a 128-bit register.
"""
@inline function hsum_128_i16(v::SIMD.Vec{8,Int16})
    s = Int32(0)
    for i in 1:8
        s += Int32(v[i])
    end
    return s
end

# ─── Aliases for Test/Fallback Compatibility ─────────────────────────

const vpmaddubsw_256 = multiply_add_pairs_256
const vpmaddubsw_128 = multiply_add_pairs_128
const _split_256 = split_256
const _combine_256 = combine_256
const _broadcast_256 = broadcast_256
const _zero_256 = zero_256
const _zero_256_int8 = zero_256_int8

end # module Intrinsics_x86_64
