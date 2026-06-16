using Test
using TriLite

@testset "Model Tests" begin

    @testset "RoPE" begin
        q = Float32[1.0, 0.0, 0.0, 0.0]
        k = Float32[0.0, 1.0, 0.0, 0.0]
        q_orig = copy(q)
        k_orig = copy(k)
        apply_rope!(q, k, 5, 4)
        @test q != q_orig
        @test k != k_orig
        @test abs(sum(q.^2) - sum(q_orig.^2)) < 1e-6
        @test abs(sum(k.^2) - sum(k_orig.^2)) < 1e-6
    end

    @testset "RoPE single head" begin
        vec = Float32[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
        vec_orig = copy(vec)
        apply_rope_single!(vec, 5, 4)
        @test vec != vec_orig
        @test abs(sum(vec.^2) - sum(vec_orig.^2)) < 1e-6
    end

    @testset "Sampling" begin
        @test greedy_sample(Float32[1.0, 2.0, 3.0, 0.5]) == 3
        for _ in 1:10
            @test topk_sample(Float32[1.0, 2.0, 3.0, 0.5], 1) == 3
        end
        for _ in 1:10
            @test topp_sample(Float32[10.0, 1.0, 1.0, 1.0], 0.1f0) == 1
        end
        for _ in 1:10
            @test sample_token(Float32[1.0, 2.0, 3.0, 0.5], temperature=0.1f0, top_k=1) == 3
        end
    end

    @testset "BitLinear" begin
        bl = BitLinear(Int8[-1, 0, 1, -1, 0, 1], (2, 3), 0.5f0, nothing)
        @test bl.weight_shape == (2, 3)
        @test bl.scale == 0.5f0
        @test bl.bias === nothing
    end

    @testset "KVCache" begin
        cache = KVCache(2, 128, 64)
        @test cache.current_len == 0
        @test cache.max_len == 128
        @test size(cache.keys) == (2, 128, 64)
        @test size(cache.values) == (2, 128, 64)
    end

    @testset "BitNetConfig" begin
        config = BitNetConfig(
            hidden_dim=2048, num_heads=32, num_kv_heads=32, head_dim=64,
            num_layers=26, ffn_dim=5632, vocab_size=128256, max_seq_len=4096,
            norm_eps=Float32(1e-6), rotary_dim=64, rope_base=10000.0f0
        )
        @test config.hidden_dim == 2048
        @test config.num_heads == 32
        @test config.head_dim == 64
        @test config.num_kv_heads == 32
        @test config.rope_base == 10000.0f0
    end

    @testset "Forward Pass (no GQA)" begin
        hidden_dim = 32
        ffn_dim = 64
        vocab_size = 100
        num_layers = 1
        num_heads = 4
        head_dim = 8
        num_kv_heads = 4

        config = BitNetConfig(
            hidden_dim=hidden_dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, num_layers=num_layers, ffn_dim=ffn_dim,
            vocab_size=vocab_size, max_seq_len=32, norm_eps=Float32(1e-6),
            rotary_dim=head_dim, rope_base=10000.0f0
        )

        kv_dim = num_kv_heads * head_dim
        layers = TransformerLayer[]
        for _ in 1:num_layers
            wq = BitLinear(vec(rand(Int8, hidden_dim, hidden_dim)), (hidden_dim, hidden_dim), 0.1f0, nothing)
            wk = BitLinear(vec(rand(Int8, kv_dim, hidden_dim)), (kv_dim, hidden_dim), 0.1f0, nothing)
            wv = BitLinear(vec(rand(Int8, kv_dim, hidden_dim)), (kv_dim, hidden_dim), 0.1f0, nothing)
            wo = BitLinear(vec(rand(Int8, hidden_dim, hidden_dim)), (hidden_dim, hidden_dim), 0.1f0, nothing)
            w_gate = BitLinear(vec(rand(Int8, ffn_dim, hidden_dim)), (ffn_dim, hidden_dim), 0.1f0, nothing)
            w_up = BitLinear(vec(rand(Int8, ffn_dim, hidden_dim)), (ffn_dim, hidden_dim), 0.1f0, nothing)
            w_down = BitLinear(vec(rand(Int8, hidden_dim, ffn_dim)), (hidden_dim, ffn_dim), 0.1f0, nothing)
            att_norm = ones(Float32, hidden_dim)
            ffn_norm = ones(Float32, hidden_dim)
            push!(layers, TransformerLayer(wq, wk, wv, wo, w_gate, w_up, w_down, att_norm, ffn_norm))
        end

        tok_emb = rand(Float16, vocab_size, hidden_dim)
        output_norm = ones(Float32, hidden_dim)
        lm_head = rand(Float16, vocab_size, hidden_dim)
        kv_cache = KVCache(num_layers, 32, kv_dim)
        model = BitNetModel(config, layers, tok_emb, output_norm, lm_head, kv_cache, nothing)

        logits = zeros(Float32, vocab_size)
        bitnet_forward_decode!(logits, model, 1, 0)
        @test !all(logits .== 0.0f0)
        @test length(logits) == vocab_size
    end

    @testset "Forward Pass (GQA)" begin
        hidden_dim = 32
        ffn_dim = 64
        vocab_size = 100
        num_layers = 1
        num_heads = 4
        num_kv_heads = 2
        head_dim = 8

        config = BitNetConfig(
            hidden_dim=hidden_dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, num_layers=num_layers, ffn_dim=ffn_dim,
            vocab_size=vocab_size, max_seq_len=32, norm_eps=Float32(1e-6),
            rotary_dim=head_dim, rope_base=10000.0f0
        )

        kv_dim = num_kv_heads * head_dim
        layers = TransformerLayer[]
        for _ in 1:num_layers
            wq = BitLinear(vec(rand(Int8, hidden_dim, hidden_dim)), (hidden_dim, hidden_dim), 0.1f0, nothing)
            wk = BitLinear(vec(rand(Int8, kv_dim, hidden_dim)), (kv_dim, hidden_dim), 0.1f0, nothing)
            wv = BitLinear(vec(rand(Int8, kv_dim, hidden_dim)), (kv_dim, hidden_dim), 0.1f0, nothing)
            wo = BitLinear(vec(rand(Int8, hidden_dim, hidden_dim)), (hidden_dim, hidden_dim), 0.1f0, nothing)
            w_gate = BitLinear(vec(rand(Int8, ffn_dim, hidden_dim)), (ffn_dim, hidden_dim), 0.1f0, nothing)
            w_up = BitLinear(vec(rand(Int8, ffn_dim, hidden_dim)), (ffn_dim, hidden_dim), 0.1f0, nothing)
            w_down = BitLinear(vec(rand(Int8, hidden_dim, ffn_dim)), (hidden_dim, ffn_dim), 0.1f0, nothing)
            att_norm = ones(Float32, hidden_dim)
            ffn_norm = ones(Float32, hidden_dim)
            push!(layers, TransformerLayer(wq, wk, wv, wo, w_gate, w_up, w_down, att_norm, ffn_norm))
        end

        tok_emb = rand(Float16, vocab_size, hidden_dim)
        output_norm = ones(Float32, hidden_dim)
        lm_head = rand(Float16, vocab_size, hidden_dim)
        kv_cache = KVCache(num_layers, 32, kv_dim)
        model = BitNetModel(config, layers, tok_emb, output_norm, lm_head, kv_cache, nothing)

        logits = zeros(Float32, vocab_size)
        bitnet_forward_decode!(logits, model, 1, 0)
        @test !all(logits .== 0.0f0)
        @test length(logits) == vocab_size
    end
end

println("All model tests passed!")
