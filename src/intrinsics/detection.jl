"""
CPU feature detection for SIMD dispatch.
"""
module FeatureDetect

export has_avx2, has_avx512, has_avx512_vbmi, has_neon, has_sve

function detect_features()
    features = (
        avx2 = false,
        avx512 = false,
        avx512_vbmi = false,
        neon = false,
        sve = false,
    )

    @static if Sys.ARCH == :x86_64
        # Use LLVM intrinsic to query CPU features
        features = (
            avx2 = _check_feature(5),
            avx512 = _check_feature(42),
            avx512_vbmi = _check_feature(33),
            avx512_vnni = _check_feature(52),
        )
    elseif Sys.ARCH == :aarch64
        features = (
            neon = true,  # Always available on aarch64
            sve = _check_arm_feature(:sve),
        )
    end

    return features
end

# x86 feature check via cpuid
function _check_feature(leaf::Int)
    try
        # Julia 1.12+: use ccall to cpuid
        regs = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
        # cpuid with leaf
        @static if Sys.ARCH == :x86_64
            c = ccall(:jl_cpuid, Cint, (Cint, Cint), leaf, 0)
            return c != 0
        end
    catch
        return false
    end
    return false
end

# ARM feature check (placeholder — actual impl uses /proc/cpuinfo or hwcap)
function _check_arm_feature(feature::Symbol)
    try
        # Linux: check hwcap
        if Sys.islinux()
            # HWCAP2_SVE = 1 << 31
            io = open("/proc/self/auxv")
            content = read(io, String)
            close(io)
            return occursin("sve", content)
        end
    catch
    end
    return false
end

# Cache the result
const FEATURES = Ref{NamedTuple}(detect_features())

has_avx2() = FEATURES[].avx2
has_avx512() = FEATURES[].avx512
has_avx512_vbmi() = FEATURES[].avx512_vbmi
has_neon() = FEATURES[].neon
has_sve() = FEATURES[].sve

end # module FeatureDetect
