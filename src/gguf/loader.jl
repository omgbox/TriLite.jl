"""
GGUF v3 file loader for BitNet models.
"""

struct GGUFTensorInfo
    name::String
    n_dims::Int
    shape::Vector{Int}
    dtype::Int
    offset::Int
end

struct GGUFFile
    path::String
    metadata::Dict{String,Any}
    tensors::Vector{GGUFTensorInfo}
    tensor_map::Dict{String,GGUFTensorInfo}
    data_offset::Int
    data::Vector{UInt8}
end

const GGUF_MAGIC = 0x46554747
const GGUF_VERSION = 3

# GGUF v3 metadata types (from ggml source)
const GTYPE_UINT8   = 0
const GTYPE_INT8    = 1
const GTYPE_UINT16  = 2
const GTYPE_INT16   = 3
const GTYPE_UINT32  = 4
const GTYPE_INT32   = 5
const GTYPE_FLOAT32 = 6
const GTYPE_BOOL    = 7
const GTYPE_STRING  = 8
const GTYPE_ARRAY   = 9
const GTYPE_UINT64  = 10
const GTYPE_INT64   = 11
const GTYPE_FLOAT64 = 12

function load_gguf(path::String; use_mmap::Bool=true)
    if !isfile(path)
        error("GGUF file not found: $path")
    end

    io = open(path, "r")

    magic = read(io, UInt32)
    if magic != GGUF_MAGIC
        error("Invalid GGUF file")
    end

    version = read(io, UInt32)
    if version != GGUF_VERSION
        @warn "GGUF version $version (expected $GGUF_VERSION)"
    end

    tensor_count = Int(read(io, UInt64))
    kv_count = Int(read(io, UInt64))

    metadata = _parse_metadata(io, kv_count)
    tensors = _parse_tensors(io, tensor_count)

    data_offset = position(io)
    data_offset = cld(data_offset, 32) * 32

    close(io)

    # Read tensor data into memory
    data = if use_mmap
        f = open(path, "r")
        seek(f, data_offset)
        filesize_data = filesize(f) - data_offset
        buf = Vector{UInt8}(undef, filesize_data)
        read!(f, buf)
        close(f)
        buf
    else
        UInt8[]
    end

    tensor_map = Dict{String,GGUFTensorInfo}(t.name => t for t in tensors)
    return GGUFFile(path, metadata, tensors, tensor_map, data_offset, data)
end

function _parse_metadata(io::IO, count::Int)
    metadata = Dict{String,Any}()
    for _ in 1:count
        key = _read_gguf_string(io)
        value_type = read(io, UInt32)
        value = _read_value(io, value_type)
        metadata[key] = value
    end
    return metadata
end

function _read_gguf_string(io::IO)
    len = read(io, UInt64)
    return String(read(io, len))
end

function _read_value(io::IO, tc::UInt32)
    tc == GTYPE_UINT8   && return read(io, UInt8)
    tc == GTYPE_INT8    && return read(io, Int8)
    tc == GTYPE_UINT16  && return read(io, UInt16)
    tc == GTYPE_INT16   && return read(io, Int16)
    tc == GTYPE_UINT32  && return read(io, UInt32)
    tc == GTYPE_INT32   && return read(io, Int32)
    tc == GTYPE_FLOAT32 && return read(io, Float32)
    tc == GTYPE_BOOL    && return read(io, UInt8) != 0
    tc == GTYPE_UINT64  && return read(io, UInt64)
    tc == GTYPE_INT64   && return read(io, Int64)
    tc == GTYPE_FLOAT64 && return read(io, Float64)
    if tc == GTYPE_STRING
        return _read_gguf_string(io)
    end
    if tc == GTYPE_ARRAY
        arr_type = read(io, UInt32)
        arr_len = Int(read(io, UInt64))
        arr = Vector{Any}(undef, arr_len)
        for i in 1:arr_len
            arr[i] = _read_value(io, arr_type)
        end
        return arr
    end
    error("Unknown GGUF type: $tc")
end

function _parse_tensors(io::IO, count::Int)
    tensors = Vector{GGUFTensorInfo}(undef, count)
    for i in 1:count
        name = _read_gguf_string(io)
        n_dims = Int(read(io, UInt32))
        shape = [Int(read(io, UInt64)) for _ in 1:n_dims]
        dtype = Int(read(io, UInt32))
        offset = Int(read(io, UInt64))
        tensors[i] = GGUFTensorInfo(name, n_dims, shape, dtype, offset)
    end
    return tensors
end

function load_tensor(gguf::GGUFFile, name::String)
    tensor_info = get(gguf.tensor_map, name, nothing)
    tensor_info === nothing && error("Tensor not found: $name")

    total_elements = prod(tensor_info.shape)
    byte_offset = tensor_info.offset
    # Calculate data size for this tensor
    elem_size = _dtype_size(tensor_info.dtype, total_elements)
    start_idx = byte_offset + 1
    end_idx = byte_offset + elem_size
    buf = IOBuffer(view(gguf.data, start_idx:end_idx))
    data = _read_tensor_data(buf, tensor_info.dtype, total_elements)
    if length(tensor_info.shape) == 2
        return permutedims(reshape(data, tensor_info.shape...))
    else
        return reshape(data, tensor_info.shape...)
    end
end

function _dtype_size(dtype::Int, n::Int)
    if dtype == 0      # F32
        return n * 4
    elseif dtype == 1  # F16
        return n * 2
    elseif dtype == 8  # Q8_0
        return (n ÷ 32) * (4 + 32)  # scale + 32 bytes per block
    elseif dtype == 36  # I2_S
        return n ÷ 4
    else
        return n * 4  # fallback
    end
end

function _read_tensor_data(io::IO, dtype::Int, n::Int)
    if dtype == 0      # F32
        data = Vector{Float32}(undef, n)
        read!(io, data)
        return data
    elseif dtype == 1  # F16
        raw = Vector{UInt16}(undef, n)
        read!(io, raw)
        return reinterpret(Float16, raw)
    elseif dtype == 8  # Q8_0
        block_size = 32
        n_blocks = n ÷ block_size
        data = Vector{Float32}(undef, n)
        for b in 1:n_blocks
            scale = read(io, Float32)
            for i in 1:block_size
                q = read(io, Int8)
                data[(b-1)*block_size + i] = q * scale
            end
        end
        return data
    elseif dtype == 36  # I2_S (BitNet ternary: 2-bit packed, 4 elements per byte)
        return _read_i2_s(io, n)
    else
        error("Unsupported GGUF tensor dtype: $dtype. The model may use bitnet-specific quantization.")
    end
end

"""
Read I2_S (Int2 with Scale) packed ternary tensor data.
Format: 2 bits per element, packed 4 per byte.
Mapping: 0→-1, 1→0, 2→+1
"""
function _read_i2_s(io::IO, n::Int)
        n_packed = n ÷ 4
        packed = Vector{UInt8}(undef, n_packed)
        read!(io, packed)
    data = Vector{Float32}(undef, n)

    @inbounds for i in 1:n_packed
        b = packed[i]
        base = (i - 1) * 4
        # Extract 4 two-bit values from each byte (MSB first)
        v0 = (b >> 6) & 0x03  # bits 7-6
        v1 = (b >> 4) & 0x03  # bits 5-4
        v2 = (b >> 2) & 0x03  # bits 3-2
        v3 = b & 0x03         # bits 1-0
        # Map: 0→-1, 1→0, 2→+1, 3→0 (shouldn't appear in valid data)
        data[base + 1] = v0 == 0 ? -1.0f0 : v0 == 2 ? 1.0f0 : 0.0f0
        data[base + 2] = v1 == 0 ? -1.0f0 : v1 == 2 ? 1.0f0 : 0.0f0
        data[base + 3] = v2 == 0 ? -1.0f0 : v2 == 2 ? 1.0f0 : 0.0f0
        data[base + 4] = v3 == 0 ? -1.0f0 : v3 == 2 ? 1.0f0 : 0.0f0
    end

    return data
end

const I2S_LUT = Int8[-1, 0, 1, 0]  # 2-bit value → ternary

function load_tensor_i8(gguf::GGUFFile, name::String)
    tensor_info = get(gguf.tensor_map, name, nothing)
    tensor_info === nothing && error("Tensor not found: $name")
    tensor_info.dtype != 36 && error("Tensor $name is not I2_S (dtype=36)")

    total_elements = prod(tensor_info.shape)
    n_packed = total_elements ÷ 4
    start_idx = tensor_info.offset + 1
    end_idx = tensor_info.offset + n_packed
    packed = @view gguf.data[start_idx:end_idx]

    data = Vector{Int8}(undef, total_elements)
    @inbounds for i in 1:n_packed
        b = packed[i]
        base = (i - 1) << 2
        data[base + 1] = I2S_LUT[((b >> 6) & 0x03) + 1]
        data[base + 2] = I2S_LUT[((b >> 4) & 0x03) + 1]
        data[base + 3] = I2S_LUT[((b >> 2) & 0x03) + 1]
        data[base + 4] = I2S_LUT[(b & 0x03) + 1]
    end

    if length(tensor_info.shape) == 2
        return permutedims(reshape(data, tensor_info.shape...))
    else
        return reshape(data, tensor_info.shape...)
    end
end

function extract_config(gguf::GGUFFile)
    meta = gguf.metadata
    arch = get(meta, "general.architecture", "bitnet-b1.58")
    prefix = "$arch."

    return (
        architecture = arch,
        hidden_dim = Int(get(meta, "$(prefix)embedding_length", 2048)),
        num_heads = Int(get(meta, "$(prefix)attention.head_count", 32)),
        num_layers = Int(get(meta, "$(prefix)block_count", 26)),
        ffn_dim = Int(get(meta, "$(prefix)feed_forward_length", 5632)),
        vocab_size = Int(get(meta, "$(prefix)vocab_size", 128256)),
        max_seq_len = Int(get(meta, "$(prefix)context_length", 4096)),
        norm_eps = Float32(get(meta, "$(prefix)attention.layer_norm_rms_epsilon", 1e-5)),
        rope_freq_base = Float32(get(meta, "$(prefix)rope.freq_base", 10000.0)),
        rope_dim = Int(get(meta, "$(prefix)rope.dimension_count", 128)),
        num_kv_heads = Int(get(meta, "$(prefix)attention.head_count_kv", 32)),
    )
end

function get_tensor_names(gguf::GGUFFile)
    return [t.name for t in gguf.tensors]
end

function has_tensor(gguf::GGUFFile, name::String)
    return any(t.name == name for t in gguf.tensors)
end

export load_gguf, GGUFFile, GGUFTensorInfo
export load_tensor, load_tensor_i8, extract_config, get_tensor_names, has_tensor
