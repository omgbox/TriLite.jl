using TriLite

println("Testing fixed GGUF loader...")
println("=" ^ 60)

path = "models/ggml-model-i2_s.gguf"

gguf = load_gguf(path)
println("Loaded: $(basename(gguf.path))")
println("Tensors: $(length(gguf.tensors))")
println("Metadata keys: $(length(gguf.metadata))")
println("Data offset: $(gguf.data_offset)")

# Print metadata (skip tokenizer arrays for brevity)
println("\nMetadata:")
for (k, v) in sort(collect(gguf.metadata), by=first)
    v_str = string(v)
    if v_str isa String && length(v_str) > 80
        v_str = v_str[1:80] * "..."
    end
    if v isa Vector && length(v) > 10
        println("  $k = [$(length(v)) elements]")
    else
        println("  $k = $v_str")
    end
end

# Print tensor info
println("\nTensor names (all $(length(gguf.tensors))):")
for t in gguf.tensors
    println("  $(t.name)  shape=$(t.shape)  ggml_dtype=$(t.dtype)")
end

# Test extract_config
println("\nModel config:")
config = extract_config(gguf)
for (k, v) in pairs(config)
    println("  $k = $v")
end

println("\n" * "=" ^ 60)
println("GGUF loader working!")
