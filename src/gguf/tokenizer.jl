# GPT-2 byte-to-unicode mapping and BPE tokenizer for TriLite.
# BitTokenizer struct is defined in types.jl.

# Build byte-to-unicode mapping (same as GPT-2's bytes_to_unicode())
function _build_byte_encoding()
    bs = Int[]
    append!(bs, 33:126)   # '!' to '~'
    append!(bs, 161:172)  # '¡' to '¬'
    append!(bs, 174:255)  # '®' to 'ÿ'
    cs = copy(bs)
    n = 0
    for b in 0:255
        if !(b in bs)
            push!(bs, b)
            push!(cs, 256 + n)
            n += 1
        end
    end
    return Dict(zip(bs, cs)), Dict(zip(cs, bs))
end

const BYTE_TO_UNICODE, UNICODE_TO_BYTE = _build_byte_encoding()

function encode_tokenizer(text::String, tok::BitTokenizer; add_bos::Bool=true)
    tokens = Int[]
    if add_bos
        push!(tokens, tok.bos_token)
    end
    for byte in codeunits(text)
        unicode_pt = get(BYTE_TO_UNICODE, Int(byte), Int(byte))
        byte_str = string(Char(unicode_pt))
        if haskey(tok.vocab, byte_str)
            push!(tokens, tok.vocab[byte_str])
        else
            push!(tokens, tok.bos_token)
        end
    end
    if isempty(tok.merge_map)
        return tokens
    end
    return _bpe_merge(tokens, tok)
end

function _heap_push!(heap::Vector{Tuple{Int,Int}}, val::Tuple{Int,Int})
    push!(heap, val)
    i = length(heap)
    while i > 1
        p = i >> 1
        heap[i][1] >= heap[p][1] && break
        heap[i], heap[p] = heap[p], heap[i]
        i = p
    end
end

function _heap_pop!(heap::Vector{Tuple{Int,Int}})
    val = heap[1]
    heap[1] = heap[end]
    pop!(heap)
    n = length(heap)
    i = 1
    while true
        left = i << 1
        right = left | 1
        smallest = i
        if left <= n && heap[left][1] < heap[smallest][1]
            smallest = left
        end
        if right <= n && heap[right][1] < heap[smallest][1]
            smallest = right
        end
        smallest == i && break
        heap[i], heap[smallest] = heap[smallest], heap[i]
        i = smallest
    end
    return val
end

function _bpe_merge(ids::Vector{Int}, tok::BitTokenizer)
    n = length(ids)
    n <= 1 && return ids

    prev = collect(0:(n-1))
    nxt = collect(2:(n+1))
    active = trues(n)
    ids_cur = copy(ids)

    heap = Tuple{Int,Int}[]
    for i in 1:(n-1)
        pair = (ids_cur[i], ids_cur[i+1])
        rank = get(tok.merge_rank, pair, typemax(Int))
        if rank < typemax(Int)
            _heap_push!(heap, (rank, i))
        end
    end

    while !isempty(heap)
        rank, pos = _heap_pop!(heap)
        nxt_pos = nxt[pos]
        nxt_pos > n && continue
        if !active[pos] || !active[nxt_pos]
            continue
        end
        right = nxt_pos
        merged_id = get(tok.merge_map, (ids_cur[pos], ids_cur[right]), 0)
        merged_id == 0 && continue

        ids_cur[pos] = merged_id
        active[right] = false
        nxt[pos] = nxt[right]
        if nxt[pos] <= n
            prev[nxt[pos]] = pos
        end

        if prev[pos] >= 1
            left = prev[pos]
            pair = (ids_cur[left], ids_cur[pos])
            r = get(tok.merge_rank, pair, typemax(Int))
            if r < typemax(Int)
                _heap_push!(heap, (r, left))
            end
        end
        if nxt[pos] <= n
            pair = (ids_cur[pos], ids_cur[nxt[pos]])
            r = get(tok.merge_rank, pair, typemax(Int))
            if r < typemax(Int)
                _heap_push!(heap, (r, pos))
            end
        end
    end

    result = Int[]
    i = 1
    while i <= n
        if active[i]
            push!(result, ids_cur[i])
        end
        nxt_i = nxt[i]
        i = (nxt_i > n) ? n + 1 : nxt_i
    end
    return result
end

function is_special(tok::BitTokenizer, token_id::Int)
    return token_id == tok.bos_token || token_id == tok.eos_token ||
           token_id in values(tok.special_tokens)
end

function decode_tokenizer(tok::BitTokenizer, tokens::Vector{Int}; skip_special::Bool=true)
    parts = String[]
    for token_id in tokens
        if skip_special && is_special(tok, token_id)
            continue
        end
        if haskey(tok.vocab_r, token_id)
            push!(parts, tok.vocab_r[token_id])
        else
            push!(parts, "<unk>")
        end
    end
    # Convert GPT-2 unicode byte encoding back to raw bytes
    combined = join(parts)
    bytes = UInt8[]
    for ch in combined
        cp = Int(ch)
        byte_val = get(UNICODE_TO_BYTE, cp, cp)
        if byte_val > 255
            byte_val = 32  # fallback for unmappable
        end
        push!(bytes, UInt8(byte_val))
    end
    return String(bytes)
end
