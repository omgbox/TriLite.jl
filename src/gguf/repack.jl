"""
Weight repacking for SIMD layout.
"""

"""
    repack_mad!(out, W)

Repack ternary weights for MAD kernel.
"""
function repack_mad!(out::AbstractVector{Int8}, W::AbstractVector{Int8})
    n = length(W)
    padded_n = cld(n, 32) * 32
    resize!(out, padded_n)
    fill!(out, Int8(0))
    out[1:n] .= W
    return out
end

function repack_mad(W::AbstractVector{Int8})
    out = Int8[]
    repack_mad!(out, W)
    return out
end

"""
    repack_lut!(tables, W, out_features, in_features)

Repack ternary weights into LUT tables.
"""
function repack_lut!(tables::AbstractVector{UInt8}, W::AbstractVector{Int8},
                     out_features::Int, in_features::Int)
    group_size = 3
    num_groups = cld(in_features, group_size)
    table_size_per_row = 32
    resize!(tables, out_features * table_size_per_row)

    for row in 1:out_features
        table_offset = (row - 1) * table_size_per_row + 1
        for g in 1:num_groups
            base_idx = (row - 1) * in_features + (g - 1) * group_size + 1
            w0 = base_idx <= length(W) ? W[base_idx] : Int8(0)
            w1 = base_idx + 1 <= length(W) ? W[base_idx + 1] : Int8(0)
            w2 = base_idx + 2 <= length(W) ? W[base_idx + 2] : Int8(0)

            for pattern in 0:26
                x0 = _decode_ternary_repack(pattern, 0)
                x1 = _decode_ternary_repack(pattern, 1)
                x2 = _decode_ternary_repack(pattern, 2)
                partial = w0 * x0 + w1 * x1 + w2 * x2
                tables[table_offset + pattern] = UInt8(partial + 3)
            end
        end
    end
    return tables
end

@inline function _decode_ternary_repack(pattern::Int, bit_pos::Int)
    bits = (pattern >> (bit_pos * 2)) & 0x03
    return bits == 0 ? Int8(-1) : (bits == 1 ? Int8(0) : Int8(1))
end

"""
    get_cache_path(path, suffix) -> String

Generate cache file path for repacked weights.
"""
function get_cache_path(path::String, suffix::String)
    dir = dirname(path)
    base = basename(path)
    name = splitext(base)[1]
    return joinpath(dir, "$(name)_$(suffix).bin")
end

"""
    save_repacked(path, data)

Save repacked weights to disk.
"""
function save_repacked(path::String, data::AbstractArray)
    open(path, "w") do io
        write(io, data)
    end
end

"""
    load_repacked(path, ::Type{T}, shape) -> Array{T}

Load repacked weights from disk.
"""
function load_repacked(path::String, ::Type{T}, shape::Tuple) where T
    if !isfile(path)
        return nothing
    end
    data = open(path, "r") do io
        read(io, T, prod(shape))
    end
    return reshape(data, shape...)
end
