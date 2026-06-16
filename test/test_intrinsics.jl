using Test
using SIMD

# Include the intrinsics modules
include("../src/intrinsics/detection.jl")
using .FeatureDetect

@static if Sys.ARCH == :x86_64
    include("../src/intrinsics/x86_64.jl")
    using .Intrinsics_x86_64
elseif Sys.ARCH == :aarch64
    include("../src/intrinsics/aarch64.jl")
    using .Intrinsics_aarch64
else
    include("../src/intrinsics/fallback.jl")
    using .Intrinsics_Fallback
end

@testset "Intrinsics Tests" begin

    @testset "pshufb_128" begin
        a = SIMD.Vec{16,UInt8}((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
        b = SIMD.Vec{16,UInt8}((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
        result = pshufb_128(a, b)
        @test all(result == a)

        b_rev = SIMD.Vec{16,UInt8}((15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
        result_rev = pshufb_128(a, b_rev)
        @test all(result_rev == SIMD.Vec{16,UInt8}((15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)))
    end

    @testset "pshufb_256" begin
        vals = ntuple(i -> UInt8(i-1), Val(32))
        a = SIMD.Vec{32,UInt8}(vals)
        b = SIMD.Vec{32,UInt8}(vals)
        result = pshufb_256(a, b)
        @test all(result == a)
    end

    @testset "vpmaddubsw_256" begin
        a_vals = ntuple(i -> UInt8(i), Val(32))
        b_vals = ntuple(i -> Int8(1), Val(32))
        a = SIMD.Vec{32,UInt8}(a_vals)
        b = SIMD.Vec{32,Int8}(b_vals)

        result = vpmaddubsw_256(a, b)

        @test result[1] == Int16(1) + Int16(2)
        @test result[2] == Int16(3) + Int16(4)
        @test result[3] == Int16(5) + Int16(6)
    end

    @testset "vpmaddubsw_128" begin
        a_vals = ntuple(i -> UInt8(i), Val(16))
        b_vals = ntuple(i -> Int8(1), Val(16))
        a = SIMD.Vec{16,UInt8}(a_vals)
        b = SIMD.Vec{16,Int8}(b_vals)

        result = vpmaddubsw_128(a, b)

        @test result[1] == Int16(1) + Int16(2)
        @test result[2] == Int16(3) + Int16(4)
    end

    @testset "Helper Functions" begin
        @testset "_split_256" begin
            vals = ntuple(i -> UInt8(i), Val(32))
            v = SIMD.Vec{32,UInt8}(vals)
            lo, hi = _split_256(v)
            @test all(lo == SIMD.Vec{16,UInt8}((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)))
            @test all(hi == SIMD.Vec{16,UInt8}((17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)))
        end

        @testset "_combine_256" begin
            lo = SIMD.Vec{16,UInt8}((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
            hi = SIMD.Vec{16,UInt8}((17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32))
            result = _combine_256(lo, hi)
            expected = SIMD.Vec{32,UInt8}(ntuple(i -> UInt8(i), Val(32)))
            @test all(result == expected)
        end

        @testset "_broadcast_256" begin
            v = _broadcast_256(UInt8(42))
            @test all(v[i] == 42 for i in 1:32)

            v_int8 = _broadcast_256(Int8(-7))
            @test all(v_int8[i] == -7 for i in 1:32)
        end

        @testset "_zero_256" begin
            v = _zero_256()
            @test all(v[i] == 0 for i in 1:32)
        end
    end

    @testset "Load/Store" begin
        @testset "load_256 / store_256!" begin
            data = collect(UInt8(1):UInt8(32))
            ptr = pointer(data)
            loaded = load_256(ptr)
            @test all(loaded == SIMD.Vec{32,UInt8}(ntuple(i -> UInt8(i), Val(32))))

            store_data = zeros(UInt8, 32)
            store_256!(pointer(store_data), loaded)
            @test store_data == collect(UInt8(1):UInt8(32))
        end
    end

    @testset "Horizontal Sum" begin
        v_16 = SIMD.Vec{16,Int16}((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
        @test hsum_256_i16(v_16) == sum(1:16)

        v_8 = SIMD.Vec{8,Int16}((1, 2, 3, 4, 5, 6, 7, 8))
        @test hsum_128_i16(v_8) == sum(1:8)
    end

    @testset "Feature Detection" begin
        features = FeatureDetect.detect_features()
        @static if Sys.ARCH == :x86_64
            @test haskey(features, :avx2)
            @test haskey(features, :avx512)
        elseif Sys.ARCH == :aarch64
            @test haskey(features, :neon)
        end
    end
end

println("All intrinsics tests passed!")
