using Pkg; Pkg.activate(".")
using TriLite

model = load_model("models/ggml-model-i2_s.gguf")
output = generate(model, "Hello", max_tokens=20)
println("\n--- Generated Output ---")
println(output)
