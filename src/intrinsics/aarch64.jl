"""
ARM64 (aarch64) SIMD intrinsics for BitNet ternary operations.

Uses LLVM IR intrinsics for NEON:
- TBL: Table lookup (ARM equivalent of pshufb)
- SQABS: Signed saturating absolute value
- SMULL/UMULL: Widening multiply

Key difference from x86: ARM table lookup uses up to 4× 16-byte tables,
indexed from a single register. No pshufb limitation.
"""
module Intrinsics_aarch64

using SIMD

export tbl_128, tbl_128_4, sqabs_128, umull_128, smull_128, smlal_128,
       _split_256, _combine_256,
       _broadcast_256, _zero_256, _zero_256_int8,
       load_256, store_256!, load_128, store_128!,
       hsum_256_i16, hsum_128_i16

# ─── 128-bit Operations (NEON) ───────────────────────────────────────

"""
    tbl_128(a::SIMD.Vec{16,UInt8}, idx::SIMD.Vec{16,UInt8}) -> SIMD.Vec{16,UInt8}

NEON TBL: byte-wise table lookup using 1 table (16 bytes).
Each byte in `idx` selects from `a` using low 4 bits.
"""
function tbl_128(a::SIMD.Vec{16,UInt8}, idx::SIMD.Vec{16,UInt8})
    return ccall(llvmcall(
        """%a = bitcast <16 x i8> %0 to <16 x i8>
           %idx = bitcast <16 x i8> %1 to <16 x i8>
           %r = call <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(
               <16 x i8> %a, <16 x i8> %idx)
           ret <16 x i8> %r""",
        Tuple{SIMD.Vec{16,UInt8}, SIMD.Vec{16,UInt8}},
        SIMD.Vec{16,UInt8}
    )(a, idx)
end

"""
    tbl_128_4(a::SIMD.Vec{16,UInt8}, b::SIMD.Vec{16,UInt8},
              c::SIMD.Vec{16,UInt8}, d::SIMD.Vec{16,UInt8},
              idx::SIMD.Vec{16,UInt8}) -> SIMD.Vec{16,UInt8}

NEON TBL with 4 tables (64 bytes total). This is the ARM advantage:
one instruction can index across 4 tables at once, covering full LUT27 range.
"""
function tbl_128_4(a::SIMD.Vec{16,UInt8}, b::SIMD.Vec{16,UInt8},
                   c::SIMD.Vec{16,UInt8}, d::SIMD.Vec{16,UInt8},
                   idx::SIMD.Vec{16,UInt8})
    return ccall(llvmcall(
        """%a = bitcast <16 x i8> %0 to <16 x i8>
           %b = bitcast <16 x i8> %1 to <16 x i8>
           %c = bitcast <16 x i8> %2 to <16 x i8>
           %d = bitcast <16 x i8> %3 to <16 x i8>
           %idx = bitcast <16 x i8> %4 to <16 x i8>
           %r = call <16 x i8> @llvm.aarch64.neon.tbl4.v16i8(
               <16 x i8> %a, <16 x i8> %b, <16 x i8> %c, <16 x i8> %d, <16 x i8> %idx)
           ret <16 x i8> %r""",
        Tuple{SIMD.Vec{16,UInt8}, SIMD.Vec{16,UInt8},
              SIMD.Vec{16,UInt8}, SIMD.Vec{16,UInt8},
              SIMD.Vec{16,UInt8}},
        SIMD.Vec{16,UInt8}
    )(a, b, c, d, idx)
end

"""
    sqabs_128(v::SIMD.Vec{16,Int8}) -> SIMD.Vec{16,Int8}

NEON SQABS: Signed saturating absolute value.
For ternary {-1,0,+1}, this is a no-op for ±1 and keeps 0.
"""
function sqabs_128(v::SIMD.Vec{16,Int8})
    return ccall(llvmcall(
        """%v = bitcast <16 x i8> %0 to <16 x i8>
           %r = call <16 x i8> @llvm.aarch64.neon.sqabs.v16i8(<16 x i8> %v)
           ret <16 x i8> %r""",
        Tuple{SIMD.Vec{16,Int8}},
        SIMD.Vec{16,Int8}
    )(v)
end

"""
    umull_128(a::SIMD.Vec{8,UInt8}, b::SIMD.Vec{8,UInt8}) -> SIMD.Vec{8,UInt16}

NEON UMULL: Unsigned widening multiply.
"""
function umull_128(a::SIMD.Vec{8,UInt8}, b::SIMD.Vec{8,UInt8})
    return ccall(llvmcall(
        """%a = bitcast <8 x i8> %0 to <8 x i8>
           %b = bitcast <8 x i8> %1 to <8 x i8>
           %a_ext = zext <8 x i8> %a to <8 x i16>
           %b_ext = zext <8 x i8> %b to <8 x i16>
           %r = mul <8 x i16> %a_ext, %b_ext
           ret <8 x i16> %r""",
        Tuple{SIMD.Vec{8,UInt8}, SIMD.Vec{8,UInt8}},
        SIMD.Vec{8,UInt16}
    )(a, b)
end

"""
    smull_128(a::SIMD.Vec{8,Int8}, b::SIMD.Vec{8,Int8}) -> SIMD.Vec{8,Int16}

NEON SMULL: Signed widening multiply.
"""
function smull_128(a::SIMD.Vec{8,Int8}, b::SIMD.Vec{8,Int8})
    return ccall(llvmcall(
        """%a = bitcast <8 x i8> %0 to <8 x i8>
           %b = bitcast <8 x i8> %1 to <8 x i8>
           %a_ext = sext <8 x i8> %a to <8 x i16>
           %b_ext = sext <8 x i8> %b to <8 x i16>
           %r = mul <8 x i16> %a_ext, %b_ext
           ret <8 x i16> %r""",
        Tuple{SIMD.Vec{8,Int8}, SIMD.Vec{8,Int8}},
        SIMD.Vec{8,Int16}
    )(a, b)
end

"""
    smlal_128(acc::SIMD.Vec{8,Int32}, a::SIMD.Vec{8,Int16}, b::SIMD.Vec{8,Int16}) -> SIMD.Vec{8,Int32}

NEON SMLAL: Signed multiply-accumulate to wider accumulator.
"""
function smlal_128(acc::SIMD.Vec{8,Int32}, a::SIMD.Vec{8,Int16}, b::SIMD.Vec{8,Int16})
    return ccall(llvmcall(
        """%acc = bitcast <8 x i32> %0 to <8 x i32>
           %a = bitcast <8 x i16> %1 to <8 x i16>
           %b = bitcast <8 x i16> %2 to <8 x i16>
           %a_ext = sext <8 x i16> %a to <8 x i32>
           %b_ext = sext <8 x i16> %b to <8 x i32>
           %prod = mul <8 x i32> %a_ext, %b_ext
           %r = add <8 x i32> %acc, %prod
           ret <8 x i32> %r""",
        Tuple{SIMD.Vec{8,Int32}, SIMD.Vec{8,Int16}, SIMD.Vec{8,Int16}},
        SIMD.Vec{8,Int32}
    )(acc, a, b)
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

@inline function hsum_128_i16(v::SIMD.Vec{8,Int16})
    s = Int32(0)
    for i in 1:8
        s += Int32(v[i])
    end
    return s
end

@inline function hsum_256_i16(v::SIMD.Vec{16,Int16})
    s = Int32(0)
    for i in 1:16
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

end # module Intrinsics_aarch64
