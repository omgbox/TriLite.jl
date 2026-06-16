"""
Chat interface and I/O for BitNet models.
"""

"""
    load_model(path; kernel=:auto)

Load a BitNet model from a GGUF file.
"""
function load_model(path::String; kernel::Symbol=:auto)
    gguf = load_gguf(path)
    config_meta = extract_config(gguf)

    println("Loaded GGUF file: $path")
    println("  Architecture: $(config_meta.architecture)")
    println("  Hidden dim: $(config_meta.hidden_dim)")
    println("  Num heads: $(config_meta.num_heads)")
    println("  Num KV heads: $(config_meta.num_kv_heads)")
    println("  Num layers: $(config_meta.num_layers)")
    println("  FFN dim: $(config_meta.ffn_dim)")
    println("  Vocab size: $(config_meta.vocab_size)")

    config = BitNetConfig(
        hidden_dim=config_meta.hidden_dim,
        num_heads=config_meta.num_heads,
        num_kv_heads=config_meta.num_kv_heads,
        head_dim=config_meta.hidden_dim ÷ config_meta.num_heads,
        num_layers=config_meta.num_layers,
        ffn_dim=config_meta.ffn_dim,
        vocab_size=config_meta.vocab_size,
        max_seq_len=config_meta.max_seq_len,
        norm_eps=config_meta.norm_eps,
        rotary_dim=config_meta.hidden_dim ÷ config_meta.num_heads,
        rope_base=Float32(config_meta.rope_freq_base)
    )

    layers = TransformerLayer[]
    num_layers = config.num_layers
    for layer_idx in 0:(num_layers - 1)
        t0 = time()
        wq = _load_bitlinear(gguf, "blk.$layer_idx.attn_q.weight", config)
        wk = _load_bitlinear(gguf, "blk.$layer_idx.attn_k.weight", config)
        wv = _load_bitlinear(gguf, "blk.$layer_idx.attn_v.weight", config)
        wo = _load_bitlinear(gguf, "blk.$layer_idx.attn_output.weight", config)
        w_gate = _load_bitlinear(gguf, "blk.$layer_idx.ffn_gate.weight", config)
        w_up = _load_bitlinear(gguf, "blk.$layer_idx.ffn_up.weight", config)
        w_down = _load_bitlinear(gguf, "blk.$layer_idx.ffn_down.weight", config)
        att_norm = _load_norm(gguf, "blk.$layer_idx.attn_norm.weight")
        ffn_norm = _load_norm(gguf, "blk.$layer_idx.ffn_norm.weight")

        push!(layers, TransformerLayer(wq, wk, wv, wo, w_gate, w_up, w_down, att_norm, ffn_norm))
        elapsed = round(time() - t0, digits=1)
        pct = round(100 * (layer_idx + 1) / num_layers, digits=0)
        println("  Layer $layer_idx/$num_layers ($pct%) — $(elapsed)s")
    end

    tok_emb = _load_embeddings(gguf, "token_embd.weight", config)
    output_norm = _load_norm(gguf, "output_norm.weight")
    lm_head = tok_emb  # Weight tying

    kv_dim = config.num_kv_heads * config.head_dim
    kv_cache = KVCache(config.num_layers, config.max_seq_len, kv_dim)

    tokenizer = _load_tokenizer(path, gguf.metadata)

    model = BitNetModel(config, layers, tok_emb, output_norm, lm_head, kv_cache, tokenizer)
    println("Model loaded successfully!")
    return model
end

"""
    generate(model, prompt; max_tokens, temperature, top_k, top_p, stop_tokens)

Generate text from a prompt.
"""
function generate(model::BitNetModel, prompt::String;
                  max_tokens::Int=128,
                  temperature::Float32=1.0f0,
                  top_k::Int=50,
                  top_p::Float32=0.9f0,
                  stop_tokens::Vector{String}=["<|end_of_text|>"])
    tokens = encode_tokenizer(prompt, model.tokenizer, add_bos=true)
    println("Input tokens: $(length(tokens))")

    logits = zeros(Float32, model.config.vocab_size)
    bitnet_forward!(reshape(logits, :, 1), model, tokens, 0)

    next_token = sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    generated_tokens = Int[next_token]
    pos = length(tokens)

    for _ in 1:(max_tokens - 1)
        token_str = decode_tokenizer(model.tokenizer, [next_token])
        if any(occursin(stop, token_str) for stop in stop_tokens)
            break
        end

        bitnet_forward_decode!(logits, model, next_token, pos)
        pos += 1
        next_token = sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        push!(generated_tokens, next_token)
    end

    return decode_tokenizer(model.tokenizer, generated_tokens)
end

"""
    chat(model; system_prompt)

Start an interactive chat session.
"""
function chat(model::BitNetModel; system_prompt::String="You are a helpful assistant.")
    println("BitNet Chat")
    println("=" ^ 50)
    println("System: $system_prompt")
    println("Type 'quit' to exit, 'clear' to reset conversation")
    println("=" ^ 50)

    conversation = String[]

    while true
        print("\nYou: ")
        user_input = readline()
        isempty(user_input) && continue
        lowercase(user_input) == "quit" && (println("Goodbye!"); break)
        if lowercase(user_input) == "clear"
            empty!(conversation)
            println("Conversation cleared.")
            continue
        end

        push!(conversation, "User: $user_input")
        prompt = join(conversation, "\n") * "\nAssistant:"
        response = generate(model, prompt, max_tokens=256)
        response = strip(split(response, "User:")[1])
        println("\nAssistant: $response")
        push!(conversation, "Assistant: $response")
    end
end

# ─── Helpers ──────────────────────────────────────────────────────────

function _load_bitlinear(gguf, name, config)
    tensor_info = get(gguf.tensor_map, name, nothing)
    if tensor_info !== nothing && tensor_info.dtype == 36
        # I2_S tensor: load directly as Int8 ternary (fast path)
        weights_i8 = load_tensor_i8(gguf, name)
        # Compute absmean scale from ternary values
        sum_abs = Int32(0)
        for w in weights_i8
            sum_abs += abs(Int32(w))
        end
        scale = Float32(sum_abs) / Float32(length(weights_i8)) + Float32(1e-6)
        shape = size(weights_i8)
        return BitLinear(vec(weights_i8), shape, scale, nothing)
    else
        # Float16/F32 tensor: load and quantize
        weights = load_tensor(gguf, name)
        weights_f32 = Float32.(vec(weights))
        ternary = absmean_quantize(weights_f32)
        scale = Float32(mean(abs.(weights_f32)))
        shape = size(weights)
        return BitLinear(ternary, shape, scale, nothing)
    end
end

function _load_norm(gguf, name)
    weights = load_tensor(gguf, name)
    return Float32.(vec(weights))
end

function _load_embeddings(gguf, name, config)
    weights = load_tensor(gguf, name)
    if eltype(weights) == Float16
        return Matrix{Float16}(weights)
    else
        return Float16.(weights)
    end
end

function _load_tokenizer(path, metadata)
    tokens = get(metadata, "tokenizer.ggml.tokens", nothing)
    if tokens !== nothing
        vocab = Dict{String,Int}()
        for (i, t) in enumerate(tokens)
            vocab[string(t)] = i - 1
        end
        vocab_r = Dict(v => k for (k, v) in vocab)
        merges_raw = get(metadata, "tokenizer.ggml.merges", String[])
        merges = Tuple{Int,Int}[]
        for m in merges_raw
            parts = split(string(m), " ")
            if length(parts) == 2
                id0 = get(vocab, parts[1], 0)
                id1 = get(vocab, parts[2], 0)
                push!(merges, (id0, id1))
            end
        end
        special = Dict{String,Int}()
        bos = Int(get(metadata, "tokenizer.ggml.bos_token_id", 1))
        eos = Int(get(metadata, "tokenizer.ggml.eos_token_id", 2))
        return BitTokenizer(vocab, vocab_r, merges, special, bos, eos)
    end
    @warn "Tokenizer not found in GGUF metadata, using minimal tokenizer"
    vocab = Dict{String,Int}("<unk>" => 0, "<s>" => 1, "</s>" => 2)
    return BitTokenizer(vocab, Dict{Int,String}(0 => "<unk>", 1 => "<s>", 2 => "</s>"),
        Tuple{Int,Int}[], Dict{String,Int}(), 1, 2)
end
