using Test

println("Running BitNet.jl Test Suite")
println("=" ^ 60)

# Test intrinsics
println("\n1. Testing SIMD intrinsics...")
include("test_intrinsics.jl")

# Test kernels
println("\n2. Testing matmul kernels...")
include("test_kernels.jl")

# Test model
println("\n3. Testing model components...")
include("test_model.jl")

println("\n" * "=" ^ 60)
println("All tests passed!")
println("=" ^ 60)
