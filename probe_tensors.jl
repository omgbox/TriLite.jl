using TriLite

path = "models/ggml-model-i2_s.gguf"
gguf = load_gguf(path)

# Find first dtype=36 tensor
target = first(t for t in gguf.tensors if t.dtype == 36)
println("Tensor: $(target.name)  shape=$(target.shape)  dtype=$(target.dtype)")
total_elements = prod(target.shape)
println("Total elements: $total_elements")

io = open(path, "r")
seek(io, gguf.data_offset + target.offset)
raw_bytes = read(io, 4096)
close(io)

# Test 1: Interpret as FP16
println("\n=== First 20 values as FP16 ===")
for i in 1:20
    b1 = raw_bytes[2*i-1]
    b2 = raw_bytes[2*i]
    bits_le = UInt16(b1) | (UInt16(b2) << 8)
    val = reinterpret(Float16, bits_le)
    println("  [$i] 0x$(string(b1,base=16,pad=2))$(string(b2,base=16,pad=2)) -> $val")
end

# Test 2: Byte distribution
println("\n=== Byte value distribution (all 256 possible values) ===")
byte_counts = zeros(Int, 256)
for b in raw_bytes
    byte_counts[b+1] += 1
end
# Show only non-zero
for i in 1:256
    if byte_counts[i] > 0
        println("  0x$(string(i-1,base=16,pad=2)): $(byte_counts[i])")
    end
end

# Test 3: file_type metadata
file_type = get(gguf.metadata, "general.file_type", nothing)
println("\ngeneral.file_type = $file_type")

# Test 4: Data sizes
println("\n=== Tensor data sizes ===")
for t in gguf.tensors[1:min(15, length(gguf.tensors))]
    next_offset = nothing
    for t2 in gguf.tensors
        if t2.offset > t.offset
            if next_offset === nothing || t2.offset < next_offset
                next_offset = t2.offset
            end
        end
    end
    if next_offset !== nothing
        data_size = next_offset - t.offset
        n_elem = prod(t.shape)
        bpe = data_size / n_elem
        println("  $(t.name) [dtype=$(t.dtype)]: $(t.shape) data_size=$data_size bpe=$(round(bpe,digits=4))")
    end
end

# Also check the very last tensor
last_t = gguf.tensors[end]
file_size = filesize(path)
data_end = gguf.data_offset + last_t.offset + prod(last_t.shape) * sizeof(Float32)  # rough estimate
actual_data_size = file_size - gguf.data_offset
println("\nFile size: $file_size")
println("Data section starts at: $(gguf.data_offset)")
println("Data section size: $actual_data_size")
println("Total expected elements (all tensors):")
total_bytes = 0
for t in gguf.tensors
    n = prod(t.shape)
    if t.dtype == 0
        total_bytes += n * 4
    elseif t.dtype == 1
        total_bytes += n * 2
    elseif t.dtype == 36
        total_bytes += n * 2  # guess: FP16-like
    else
        total_bytes += n * 4
    end
end
println("  If dtype=36 is FP16: total = $total_bytes")
total_bytes_q2 = 0
for t in gguf.tensors
    n = prod(t.shape)
    if t.dtype == 0
        total_bytes_q2 += n * 4
    elseif t.dtype == 1
        total_bytes_q2 += n * 2
    elseif t.dtype == 36
        total_bytes_q2 += n * 2 + 4  # 2-bit packed + scale? 
    else
        total_bytes_q2 += n * 4
    end
end
println("  Actual data size: $actual_data_size")
