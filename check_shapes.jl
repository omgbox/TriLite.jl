using Pkg
Pkg.activate(".")
using TriLite

path = "models/ggml-model-i2_s.gguf"
gguf = load_gguf(path)

# Print shapes of key tensors
key_tensors = ["token_embd.weight", "output_norm.weight",
    "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight",
    "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
    "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"]

for name in key_tensors
    for t in gguf.tensors
        if t.name == name
            println("$name: shape=$(t.shape), dtype=$(t.dtype)")
            break
        end
    end
end

# Test embedding loading without permutedims
for t in gguf.tensors
    if t.name == "token_embd.weight"
        total = prod(t.shape)
        elem_size = _dtype_size(t.dtype, total)
        start_idx = t.offset + 1
        end_idx = t.offset + elem_size
        buf = IOBuffer(view(gguf.data, start_idx:end_idx))
        raw = Vector{UInt16}(undef, total)
        read!(buf, raw)
        data_f16 = reinterpret(Float16, raw)

        shape = t.shape
        println("\nRaw shape from GGUF: $shape")
        
        # Try reshape as-is (Julia column-major)
        m1 = reshape(data_f16, shape...)
        println("reshape(data, shape...) = $(size(m1))")
        println("  m1[1, 1:5] = $(Float32.(m1[1, 1:5]))")
        println("  m1[2, 1:5] = $(Float32.(m1[2, 1:5]))")
        println("  m1[:, 1] stats: mean=$(mean(Float32.(m1[:, 1]))), std=$(std(Float32.(m1[:, 1])))")
        
        # Try reshape with reversed dims  
        m2 = reshape(data_f16, reverse(shape)...)
        println("\nreshape(data, reverse(shape)) = $(size(m2))")
        println("  m2[1, 1:5] = $(Float32.(m2[1, 1:5]))")
        println("  m2[2, 1:5] = $(Float32.(m2[2, 1:5]))")
        println("  m2[:, 1] stats: mean=$(mean(Float32.(m2[:, 1]))), std=$(std(Float32.(m2[:, 1])))")
        
        # Compare: for a proper embedding, columns should look like embedding vectors
        # (high variance across dims, consistent within a token)
        # Rows should have lower variance (each dim value across tokens)
        
        # Check variance of first 100 elements of row 1 vs column 1
        r1_var = var(Float32.(m1[1, 1:100]))
        c1_var = var(Float32.(m1[1:min(100, size(m1,1)), 1]))
        println("\nm1: row1 variance (100 tokens)=$r1_var, col1 variance (100 dims)=$c1_var")
        
        r1_var2 = var(Float32.(m2[1, 1:min(100, size(m2,2))]))
        c1_var2 = var(Float32.(m2[1:min(100, size(m2,1)), 1]))
        println("m2: row1 variance (100 tokens)=$r1_var2, col1 variance (100 dims)=$c1_var2")
        break
    end
end
