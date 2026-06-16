# Kernel dispatch via trait types

@inline function matmul!(::RefKernel, out::AbstractMatrix{Float32},
                         W::AbstractMatrix{Int8}, x::AbstractMatrix{Float32},
                         scale::Float32)
    return matmul_ref!(out, W, x, scale)
end

@inline function matmul!(::MADKernel, out::AbstractMatrix{Float32},
                         W::AbstractMatrix{Int8}, x::AbstractMatrix{Float32},
                         scale::Float32)
    return matmul_mad!(out, W, x, scale)
end

@inline function matmul!(::LUTKernel, out::AbstractMatrix{Float32},
                         W::AbstractMatrix{Int8}, x::AbstractMatrix{Float32},
                         scale::Float32)
    error("LUT kernel requires pre-built LUTTables; use build_lut_tables! and matmul_lut!")
end

@inline function matmul_vec!(::RefKernel, out::AbstractVector{Float32},
                             W::AbstractMatrix{Int8}, x::AbstractVector{Float32},
                             scale::Float32)
    return matmul_ref_vec!(out, W, x, scale)
end

@inline function matmul_vec!(::MADKernel, out::AbstractVector{Float32},
                             W::AbstractMatrix{Int8}, x::AbstractVector{Float32},
                             scale::Float32)
    return matmul_mad_vec!(out, W, x, scale)
end

@inline function matmul_vec!(::LUTKernel, out::AbstractVector{Float32},
                             W::AbstractMatrix{Int8}, x::AbstractVector{Float32},
                             scale::Float32)
    error("LUT kernel requires pre-built LUTTables; use build_lut_tables! and matmul_lut_vec!")
end
