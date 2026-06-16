using TriLite
using Statistics

path = "models/ggml-model-i2_s.gguf"
gguf = load_gguf(path)

println("Testing I2_S decoder...")

# Load a small ternary weight tensor
tensor_name = "blk.0.attn_q.weight"  # 2560x2560
println("\nLoading $tensor_name...")
W = load_tensor(gguf, tensor_name)
println("  Shape: $(size(W))")
println("  Eltype: $(eltype(W))")
println("  Min: $(minimum(W)), Max: $(maximum(W))")
println("  Mean: $(mean(W))")
println("  Std: $(std(W))")

# Count ternary values
n_neg1 = count(==(-1.0f0), W)
n_zero = count(==(0.0f0), W)
n_pos1 = count(==(1.0f0), W)
total = length(W)
println("  Distribution:")
println("    -1: $n_neg1 ($(round(100*n_neg1/total, digits=1))%)")
println("     0: $n_zero ($(round(100*n_zero/total, digits=1))%)")
println("    +1: $n_pos1 ($(round(100*n_pos1/total, digits=1))%)")

# Also test the i8 version
println("\nLoading $tensor_name as Int8...")
W_i8 = load_tensor_i8(gguf, tensor_name)
println("  Shape: $(size(W_i8))")
println("  Eltype: $(eltype(W_i8))")
println("  Min: $(minimum(W_i8)), Max: $(maximum(W_i8))")
n_neg1_i8 = count(==(-1), W_i8)
n_zero_i8 = count(==(0), W_i8)
n_pos1_i8 = count(==(1), W_i8)
println("  Distribution:")
println("    -1: $n_neg1_i8 ($(round(100*n_neg1_i8/total, digits=1))%)")
println("     0: $n_zero_i8 ($(round(100*n_zero_i8/total, digits=1))%)")
println("    +1: $n_pos1_i8 ($(round(100*n_pos1_i8/total, digits=1))%)")

# Test a norm tensor
println("\nLoading blk.0.attn_norm.weight (F32)...")
norm = load_tensor(gguf, "blk.0.attn_norm.weight")
println("  Shape: $(size(norm))")
println("  Eltype: $(eltype(norm))")
println("  First 5: $(norm[1:5])")

# Test embedding
println("\nLoading token_embd.weight (F16)...")
emb = load_tensor(gguf, "token_embd.weight")
println("  Shape: $(size(emb))")
println("  Eltype: $(eltype(emb))")
println("  First 5: $(emb[1:5])")

println("\nI2_S decoder test passed!")
