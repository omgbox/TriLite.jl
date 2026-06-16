using Pkg
Pkg.activate(".")
using TriLite

hidden_dim = 32
kv_dim = 32
ffn_dim = 64
head_dim = 8
num_heads = 4
num_kv_heads = 4

att_norm = ones(Float32, hidden_dim)
ffn_norm = ones(Float32, hidden_dim)

wq = BitLinear(vec(rand(Int8, hidden_dim, hidden_dim)), (hidden_dim, hidden_dim), 0.1f0, nothing)
wk = BitLinear(vec(rand(Int8, kv_dim, hidden_dim)), (kv_dim, hidden_dim), 0.1f0, nothing)
wv = BitLinear(vec(rand(Int8, kv_dim, hidden_dim)), (kv_dim, hidden_dim), 0.1f0, nothing)
wo = BitLinear(vec(rand(Int8, hidden_dim, hidden_dim)), (hidden_dim, hidden_dim), 0.1f0, nothing)
w_gate = BitLinear(vec(rand(Int8, ffn_dim, hidden_dim)), (ffn_dim, hidden_dim), 0.1f0, nothing)
w_up = BitLinear(vec(rand(Int8, ffn_dim, hidden_dim)), (ffn_dim, hidden_dim), 0.1f0, nothing)
w_down = BitLinear(vec(rand(Int8, hidden_dim, ffn_dim)), (hidden_dim, ffn_dim), 0.1f0, nothing)

layer = TransformerLayer(wq, wk, wv, wo, w_gate, w_up, w_down, att_norm, ffn_norm)

config = BitNetConfig(
    hidden_dim=hidden_dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
    head_dim=head_dim, num_layers=1, ffn_dim=ffn_dim,
    vocab_size=100, max_seq_len=32, norm_eps=Float32(1e-6),
    rotary_dim=head_dim, rope_base=10000.0f0
)

tok_emb = rand(Float16, 100, hidden_dim)
output_norm = ones(Float32, hidden_dim)
lm_head = rand(Float16, 100, hidden_dim)
kv_cache = KVCache(1, 32, kv_dim)
model = BitNetModel(config, [layer], tok_emb, output_norm, lm_head, kv_cache, nothing)

# Step-by-step forward decode
h = Float32.(model.tok_embeddings[:, 1])
println("Step 1 - Embedding: h size = ", length(h))

# Manually do what attention_decode does
println("\n--- Step 2: attention_decode! internals ---")
hidden_dim_check = length(h)
println("hidden_dim = ", hidden_dim_check)

h_normed = similar(h)
println("rmsnorm input sizes: h_normed=", length(h_normed), " h=", length(h), " att_norm_weight=", length(att_norm))
rmsnorm!(h_normed, h, att_norm, Float32(1e-6))
println("rmsnorm OK")

q = zeros(Float32, hidden_dim)
k = zeros(Float32, kv_dim)
v = zeros(Float32, kv_dim)

# matmul_ref_vec! for q
Wq = reshape(Int8.(wq.packed_weights), wq.weight_shape...)
println("Wq size = ", size(Wq), " h_normed size = ", length(h_normed))
for row in 1:size(Wq, 1)
    acc = 0.0f0
    for kidx in 1:size(Wq, 2)
        acc += Wq[row, kidx] * h_normed[kidx]
    end
    q[row] = acc * wq.scale
end
println("q OK, size = ", length(q))

# RoPE
for h_idx in 1:num_heads
    offset = (h_idx - 1) * head_dim
    apply_rope_single!(view(q, offset+1:offset+head_dim), 0, head_dim, 10000.0f0)
end
println("RoPE OK")

# Cache store
kv_cache.current_len += 1
kv_cache.keys[1, 1, 1:kv_dim] .= k
kv_cache.values[1, 1, 1:kv_dim] .= v
println("Cache OK, current_len = ", kv_cache.current_len)

# Attention compute
println("\n--- Step 3: _compute_attention_decode ---")
println("q size = ", length(q))
println("cache keys size = ", size(kv_cache.keys))

seq_len = kv_cache.current_len
scale = 1.0f0 / sqrt(Float32(head_dim))
kv_group_size = num_heads ÷ num_kv_heads
q_r = reshape(q, head_dim, num_heads)
println("q_r size = ", size(q_r))

att_scores = zeros(Float32, num_heads, seq_len)
for h_idx in 1:num_heads, j in 1:seq_len
    kv_h = cld(h_idx, kv_group_size)
    s = 0.0f0
    for d in 1:head_dim
        s += q_r[d, h_idx] * kv_cache.keys[1, j, (kv_h-1)*head_dim+d]
    end
    att_scores[h_idx, j] = s * scale
end
println("att_scores OK, size = ", size(att_scores))

att_weights = zeros(Float32, num_heads, seq_len)
for h_idx in 1:num_heads
    softmax!(view(att_weights, h_idx, :), view(att_scores, h_idx, :))
end
println("softmax OK")

att_out = zeros(Float32, hidden_dim)
att_out_r = reshape(att_out, head_dim, num_heads)
for h_idx in 1:num_heads, d in 1:head_dim
    kv_h = cld(h_idx, kv_group_size)
    acc = 0.0f0
    for j in 1:seq_len
        acc += att_weights[h_idx, j] * kv_cache.values[1, j, (kv_h-1)*head_dim+d]
    end
    att_out_r[d, h_idx] = acc
end
println("attention output OK, size = ", length(att_out))

# Output projection
Wo = reshape(Int8.(wo.packed_weights), wo.weight_shape...)
println("Wo size = ", size(Wo), " att_out size = ", length(att_out))
out = zeros(Float32, hidden_dim)
for row in 1:size(Wo, 1)
    acc = 0.0f0
    for kidx in 1:size(Wo, 2)
        acc += Wo[row, kidx] * att_out[kidx]
    end
    out[row] = acc * wo.scale
end
println("Output projection OK")

# Residual
for i in eachindex(out)
    out[i] = h[i] + out[i]
end
println("Residual OK")

# FFN decode
println("\n--- Step 4: ffn_decode! internals ---")
h_normed2 = similar(out)
println("ffn rmsnorm sizes: h_normed2=", length(h_normed2), " out=", length(out), " ffn_norm=", length(ffn_norm))
rmsnorm!(h_normed2, out, ffn_norm, Float32(1e-6))
println("ffn rmsnorm OK")

W_gate = reshape(Int8.(w_gate.packed_weights), w_gate.weight_shape...)
W_up = reshape(Int8.(w_up.packed_weights), w_up.weight_shape...)
println("W_gate size = ", size(W_gate))
println("W_up size = ", size(W_up))

gate = zeros(Float32, ffn_dim)
up = zeros(Float32, ffn_dim)
for row in 1:ffn_dim
    acc_g = 0.0f0
    acc_u = 0.0f0
    for kidx in 1:hidden_dim
        acc_g += W_gate[row, kidx] * h_normed2[kidx]
        acc_u += W_up[row, kidx] * h_normed2[kidx]
    end
    gate[row] = acc_g * w_gate.scale
    up[row] = acc_u * w_up.scale
end
println("Gate/Up OK")

hidden_ffn = similar(gate)
for i in eachindex(gate)
    hidden_ffn[i] = relu_squared(gate[i]) * up[i]
end
println("Hidden FFN OK, size = ", length(hidden_ffn))

W_down = reshape(Int8.(w_down.packed_weights), w_down.weight_shape...)
println("W_down size = ", size(W_down))
down_out = zeros(Float32, hidden_dim)
for row in 1:hidden_dim
    acc = 0.0f0
    for kidx in 1:ffn_dim
        acc += W_down[row, kidx] * hidden_ffn[kidx]
    end
    down_out[row] = acc * w_down.scale
end
println("Down OK")

println("\nAll steps passed!")
