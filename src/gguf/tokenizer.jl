"""
Minimal BPE tokenizer for BitNet models.
BitTokenizer struct is defined in types.jl.
"""

"""
    encode_tokenizer(text::String, tok::BitTokenizer; add_bos::Bool=true) -> Vector{Int}

Encode text to token ids.
"""
function encode_tokenizer(text::String, tok::BitTokenizer; add_bos::Bool=true)
    tokens = Int[]
    if add_bos
        push!(tokens, tok.bos_token)
    end
    for char in text
        bytes = codeunits(string(char))
        for byte in bytes
            byte_str = string(Char(byte))
            if haskey(tok.vocab, byte_str)
                push!(tokens, tok.vocab[byte_str])
            else
                push!(tokens, tok.bos_token)
            end
        end
    end
    return tokens
end

"""
    decode_tokenizer(tok::BitTokenizer, tokens::Vector{Int}; skip_special::Bool=true) -> String

Decode token ids back to text.
"""
function decode_tokenizer(tok::BitTokenizer, tokens::Vector{Int}; skip_special::Bool=true)
    parts = String[]
    for token_id in tokens
        if skip_special && token_id in values(tok.special_tokens)
            continue
        end
        if haskey(tok.vocab_r, token_id)
            push!(parts, tok.vocab_r[token_id])
        else
            push!(parts, "<unk>")
        end
    end
    return join(parts)
end
