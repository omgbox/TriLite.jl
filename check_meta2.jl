using TriLite

path = "models/ggml-model-i2_s.gguf"
gguf = load_gguf(path)

println("=== Key Metadata ===")
for kv in gguf.metadata
    k, v = kv
    if !(v isa AbstractVector) && !occursin("tokenizer", k)
        println("  $k = $v")
    end
end

# Check for tensor-level scale keys
println("\n=== Tensor-level scale/quant keys ===")
for kv in gguf.metadata
    k, v = kv
    kl = lowercase(k)
    if (occursin("scale", kl) || occursin("ter_", kl) || occursin("i2_", kl) || occursin("quant", kl)) && !(v isa AbstractVector)
        println("  $k = $v")
    end
end

# Check tensor data offsets and sizes
println("\n=== Tensor data info (first 5 ternary) ===")
count = 0
for t in gguf.tensors
    if t.dtype == 36 && count < 5
        data_size = t.data_size
        expected_packed = prod(t.shape) ÷ 4
        println("  $(t.name): shape=$(t.shape), dtype=$(t.dtype), data_size=$data_size, expected_packed=$expected_packed, diff=$(data_size - expected_packed)")
        count += 1
    end
end
