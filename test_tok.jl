using Pkg; Pkg.activate(".")
using TriLite

gguf = load_gguf("models/ggml-model-i2_s.gguf")
meta = gguf.metadata
toks = meta["tokenizer.ggml.tokens"]
vocab = Dict{String,Int}(string(t) => i-1 for (i,t) in enumerate(toks))
vocab_r = Dict(v => k for (k,v) in vocab)
special = Dict{String,Int}()
tok = BitTokenizer(vocab, vocab_r, Tuple{Int,Int}[],
    special, Int(meta["tokenizer.ggml.bos_token_id"]), Int(meta["tokenizer.ggml.eos_token_id"]))

encoded = encode_tokenizer("Hello", tok)
decoded = decode_tokenizer(tok, encoded)
println("Encoded: $encoded")
println("Decoded: '$decoded'")
println("Match: $(decoded == "Hello" ? "YES" : "NO")")
