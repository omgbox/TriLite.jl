"""
Fallback scalar intrinsics for non-x86_64 / non-aarch64 platforms.
Uses plain Julia operations without SIMD intrinsics.
"""
module Intrinsics_Fallback

using SIMD

export pshufb_128, pshufb_256,
       vpmaddubsw_128, vpmaddubsw_256,
       _split_256, _combine_256,
       _broadcast_256, _zero_256, _zero_256_int8,
       load_256, store_256!, load_128, store_128!,
       hsum_256_i16, hsum_128_i16

# ─── Byte Shuffle (Scalar Reference) ─────────────────────────────────

"""
    pshufb_128(a::SIMD.Vec{16,UInt8}, b::SIMD.Vec{16,UInt8}) -> SIMD.Vec{16,UInt8}

Scalar fallback for pshufb. Each byte selects from `a` using low 4 bits of index.
"""
@inline function pshufb_128(a::SIMD.Vec{16,UInt8}, b::SIMD.Vec{16,UInt8})
    result = zeros(UInt8, 16)
    for i in 1:16
        idx = b[i] & 0x0F
        if (b[i] & 0x80) == 0
            result[i] = a[Int(idx) + 1]
        end
    end
    return SIMD.Vec(Tuple(result))
end

"""
    pshufb_256(a::SIMD.Vec{32,UInt8}, b::SIMD.Vec{32,UInt8}) -> SIMD.Vec{32,UInt8}

Scalar fallback for 256-bit byte shuffle (two 128-bit operations).
"""
@inline function pshufb_256(a::SIMD.Vec{32,UInt8}, b::SIMD.Vec{32,UInt8})
    a_lo, a_hi = _split_256(a)
    b_lo, b_hi = _split_256(b)
    r_lo = pshufb_128(a_lo, b_lo)
    r_hi = pshufb_128(a_hi, b_hi)
    return _combine_256(r_lo, r_hi)
end

# ─── Multiply-Add Pairs (Scalar Reference) ───────────────────────────

"""
    vpmaddubsw_128(a::SIMD.Vec{16,UInt8}, b::SIMD.Vec{16,Int8}) -> SIMD.Vec{8,Int16}

Scalar reference: multiply unsigned × signed adjacent pairs, sum each pair.
"""
@inline function vpmaddubsw_128(a::SIMD.Vec{16,UInt8}, b::SIMD.Vec{16,Int8})
    result = zeros(Int16, 8)
    for i in 1:8
        ai1 = Int16(a[2*(i-1)+1])
        ai2 = Int16(a[2*(i-1)+2])
        bi1 = Int16(b[2*(i-1)+1])
        bi2 = Int16(b[2*(i-1)+2])
        result[i] = ai1 * bi1 + ai2 * bi2
    end
    return SIMD.Vec(Tuple(result))
end

"""
    vpmaddubsw_256(a::SIMD.Vec{32,UInt8}, b::SIMD.Vec{32,Int8}) -> SIMD.Vec{16,Int16}

Scalar reference for 256-bit multiply-add.
"""
@inline function vpmaddubsw_256(a::SIMD.Vec{32,UInt8}, b::SIMD.Vec{32,Int8})
    result = zeros(Int16, 16)
    for i in 1:16
        ai1 = Int16(a[2*(i-1)+1])
        ai2 = Int16(a[2*(i-1)+2])
        bi1 = Int16(b[2*(i-1)+1])
        bi2 = Int16(b[2*(i-1)+2])
        result[i] = ai1 * bi1 + ai2 * bi2
    end
    return SIMD.Vec(Tuple(result))
end

# ─── Helper Functions ─────────────────────────────────────────────────

@inline function _split_256(v::SIMD.Vec{32,T}) where T
    lo = SIMD.Vec(ntuple(i -> v[i], Val(16)))
    hi = SIMD.Vec(ntuple(i -> v[i+16], Val(16)))
    return (lo, hi)
end

@inline function _combine_256(lo::SIMD.Vec{16,T}, hi::SIMD.Vec{16,T}) where T
    return SIMD.Vec(ntuple(i -> i <= 16 ? lo[i] : hi[i-16], Val(32)))
end

# ─── Horizontal Sum ───────────────────────────────────────────────────

@inline function hsum_256_i16(v::SIMD.Vec{16,Int16})
    s = Int32(0)
    for i in 1:16
        s += Int32(v[i])
    end
    return s
end

@inline function hsum_128_i16(v::SIMD.Vec{8,Int16})
    s = Int32(0)
    for i in 1:8
        s += Int32(v[i])
    end
    return s
end

# ─── Load/Store ───────────────────────────────────────────────────────

@inline function load_256(ptr::Ptr{T}) where T
    return unsafe_load(Ptr{SIMD.Vec{32,T}}(ptr))
end

@inline function store_256!(ptr::Ptr{T}, val::SIMD.Vec{32,T}) where T
    unsafe_store!(Ptr{SIMD.Vec{32,T}}(ptr), val)
end

@inline function load_128(ptr::Ptr{T}) where T
    return unsafe_load(Ptr{SIMD.Vec{16,T}}(ptr))
end

@inline function store_128!(ptr::Ptr{T}, val::SIMD.Vec{16,T}) where T
    unsafe_store!(Ptr{SIMD.Vec{16,T}}(ptr), val)
end

# ─── Broadcast ────────────────────────────────────────────────────────

@inline function _broadcast_256(val::UInt8)
    return SIMD.Vec(ntuple(_ -> val, Val(32)))
end

@inline function _broadcast_256(val::Int8)
    return SIMD.Vec(ntuple(_ -> val, Val(32)))
end

@inline function _zero_256()
    return _broadcast_256(UInt8(0))
end

@inline function _zero_256_int8()
    return _broadcast_256(Int8(0))
end

end # module Intrinsics_Fallback
