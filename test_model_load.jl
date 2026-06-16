using TriLite
using Statistics

path = "models/ggml-model-i2_s.gguf"
println("Loading model...")
t0 = time()
model = load_model(path)
println("Model loaded in $(round(time() - t0, digits=1))s")

println("\nModel config:")
println("  hidden_dim = $(model.config.hidden_dim)")
println("  num_heads = $(model.config.num_heads)")
println("  num_kv_heads = $(model.config.num_kv_heads)")
println("  head_dim = $(model.config.head_dim)")
println("  num_layers = $(model.config.num_layers)")
println("  ffn_dim = $(model.config.ffn_dim)")
println("  vocab_size = $(model.config.vocab_size)")
println("  rope_base = $(model.config.rope_base)")

# Quick smoke test: forward pass with dummy tokens
println("\nRunning forward pass with 4 dummy tokens...")
logits = zeros(Float32, model.config.vocab_size, 4)
tokens = [1, 2, 3, 4]  # dummy tokens
bitnet_forward!(logits, model, tokens, 0)
println("  Logits shape: $(size(logits))")
println("  Logits range: [$(minimum(logits)), $(maximum(logits))]")
println("  Logits mean: $(mean(logits))")

# Check logits aren't all zero
if maximum(abs.(logits)) < 1e-10
    println("  WARNING: All logits are near zero!")
else
    println("  Logits have non-zero values - forward pass works!")
    # Show top-5 predictions for last token
    last_logits = logits[:, 4]
    top5 = sortperm(last_logits, rev=true)[1:5]
    println("  Top-5 token IDs for position 4: $top5")
    println("  Top-5 logit values: $(last_logits[top5])")
end

println("\nModel load test passed!")
