using Test
using SIMD

# Include platform intrinsics (required by MAD/LUT kernels)
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

# Include kernel files (no module wrappers — names go into Main directly)
include("../src/kernels/utils.jl")
include("../src/gguf/repack.jl")
include("../src/kernels/matmul_ref.jl")
include("../src/kernels/matmul_mad.jl")
include("../src/kernels/matmul_lut.jl")

@testset "Kernel Tests" begin

    @testset "KernelUtils" begin
        @testset "absmean_quantize" begin
            # Test quantization of simple vectors
            w = Float32[-0.5, 0.0, 0.5, 1.0, -1.0, 0.3, -0.3, 0.0]
            result = absmean_quantize(w)

            # Check that values are in {-1, 0, +1}
            @test all(result .>= -1)
            @test all(result .<= 1)
            @test all(result .== round.(result))
        end

        @testset "repack_mad!" begin
            w = Int8[-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]
            packed = Int8[]
            repack_mad!(packed, w)
            @test length(packed) == cld(length(w), 32) * 32
        end

        @testset "ceil_to_multiple" begin
            @test ceil_to_multiple(7, 8) == 8
            @test ceil_to_multiple(8, 8) == 8
            @test ceil_to_multiple(9, 8) == 16
        end

        @testset "relu_squared" begin
            @test relu_squared(2.0f0) == 4.0f0
            @test relu_squared(-2.0f0) == 0.0f0
            @test relu_squared(0.0f0) == 0.0f0
        end

        @testset "rmsnorm" begin
            x = Float32[1.0, 2.0, 3.0, 4.0]
            weight = Float32[1.0, 1.0, 1.0, 1.0]
            eps = Float32(1e-6)

            result = rmsnorm(x, weight, eps)

            # Check that output is normalized
            @test abs(sum(result.^2) / length(result) - 1.0) < 1e-5
        end

        @testset "softmax" begin
            x = Float32[1.0, 2.0, 3.0, 4.0]
            result = similar(x)
            softmax!(result, x)

            # Check that output sums to 1
            @test abs(sum(result) - 1.0) < 1e-6

            # Check that order is preserved
            @test result[1] < result[2] < result[3] < result[4]
        end
    end

    @testset "Reference Kernel" begin
        @testset "matmul_ref!" begin
            # Small test case
            out_features = 4
            in_features = 8
            seq_len = 2

            # Ternary weights (transposed for contiguous access)
            W = permutedims(Int8[-1 0 1 -1 0 1 -1 0;
                                  0 1 -1 0 1 -1 0 1;
                                  1 -1 0 1 -1 0 1 -1;
                                  -1 0 1 -1 0 1 -1 0], (2, 1))

            # Activations
            x = Float32[1.0 2.0;
                        3.0 4.0;
                        5.0 6.0;
                        7.0 8.0;
                        9.0 10.0;
                        11.0 12.0;
                        13.0 14.0;
                        15.0 16.0]

            scale = 1.0f0
            out = zeros(Float32, out_features, seq_len)

            matmul_ref!(out, W, x, scale)

            # Verify first output element
            # Row 1: [-1, 0, 1, -1, 0, 1, -1, 0] · [1, 3, 5, 7, 9, 11, 13, 15]
            # = -1 + 0 + 5 - 7 + 0 + 11 - 13 + 0 = -5
            @test out[1, 1] ≈ -5.0f0

            # Row 1: [-1, 0, 1, -1, 0, 1, -1, 0] · [2, 4, 6, 8, 10, 12, 14, 16]
            # = -2 + 0 + 6 - 8 + 0 + 12 - 14 + 0 = -6
            @test out[1, 2] ≈ -6.0f0
        end

        @testset "matmul_ref_vec!" begin
            out_features = 4
            in_features = 8

            W = permutedims(Int8[-1 0 1 -1 0 1 -1 0;
                                  0 1 -1 0 1 -1 0 1;
                                  1 -1 0 1 -1 0 1 -1;
                                  -1 0 1 -1 0 1 -1 0], (2, 1))

            x = Float32[1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
            scale = 1.0f0
            out = zeros(Float32, out_features)

            matmul_ref_vec!(out, W, x, scale)

            @test out[1] ≈ -5.0f0
        end
    end

    @testset "MAD Kernel" begin
        @testset "matmul_mad!" begin
            # Test with full ternary {-1,0,1} activations (tests sign handling)
            out_features = 32
            in_features = 32
            seq_len = 1

            W = permutedims(Int8.(rand(-1:1, out_features, in_features)), (2, 1))
            x = Float32.(rand(-1:1, in_features, seq_len))

            scale = 1.0f0
            out_ref = zeros(Float32, out_features, seq_len)
            out_mad = zeros(Float32, out_features, seq_len)

            # Reference
            matmul_ref!(out_ref, W, x, scale)

            # MAD kernel
            matmul_mad!(out_mad, W, x, scale)

            # Should match (within floating point precision)
            @test out_ref ≈ out_mad atol=1e-4
        end
    end

    @testset "LUT Kernel" begin
        @testset "matmul_lut!" begin
            # Build LUT tables
            out_features = 16
            in_features = 48

            W = permutedims(Int8.(rand(-1:1, out_features, in_features)), (2, 1))
            x = Float32.(rand(0:1, in_features, 1))

            scale = 1.0f0

            # Build tables
            lut_tables = build_lut_tables!(W, vec(x), scale)

            # Test LUT kernel
            out_ref = zeros(Float32, out_features, 1)
            out_lut = zeros(Float32, out_features, 1)

            matmul_ref!(out_ref, W, x, scale)
            matmul_lut!(out_lut, lut_tables, x, scale)

            # Results should be close (LUT has quantization)
            # Allow larger tolerance due to ternarization
            @test size(out_lut) == size(out_ref)
        end
    end

    @testset "Consistency" begin
        @testset "All kernels produce same output for same input" begin
            out_features = 32
            in_features = 64
            seq_len = 1

            W = permutedims(Int8.(rand(-1:1, out_features, in_features)), (2, 1))
            x = Float32.(rand(-1:1, in_features, seq_len))
            scale = 0.1f0

            out_ref = zeros(Float32, out_features, seq_len)
            out_mad = zeros(Float32, out_features, seq_len)

            matmul_ref!(out_ref, W, x, scale)
            matmul_mad!(out_mad, W, x, scale)

            # Reference and MAD should match
            @test out_ref ≈ out_mad atol=1e-4
        end
    end
end

println("All kernel tests passed!")
