using TriLite

path = "models/ggml-model-i2_s.gguf"
gguf = load_gguf(path)

println("=== All GGUF Metadata Keys ===")
for kv in gguf.metadata
    k, v = kv
    if v isa String && length(v) > 80
        v = v[1:80] * "..."
    end
    println("  $k = $v")
end

println("\n=== Check for per-tensor scales ===")
for kv in gguf.metadata
    k, v = kv
    kl = lowercase(k)
    if occursin("scale", kl) || occursin("ter", kl) || occursin("quant", kl)
        println("  $k = $v")
    end
end
