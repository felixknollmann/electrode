# Edge-sum potential kernel, benchmark only (milestone 3; not part of the package). Same formula and loop structure
# as src/gds2itvg/kernels/numba_kernel.py; uses only Base (no registry packages).
#
# Command-line use (called from benchmarks/julia/julia_runner.py):
#   julia -t N Gds2ItvgKernel.jl IN.bin OUT.bin [REPEATS]
# IN.bin: Int64 n_points, Int64 n_edges, then Float64 points (3 x N, column-major: x,y,z per
# point), p1 (2 x E), p2 (2 x E), w (E). OUT.bin: Float64 phi (N). With REPEATS > 0 the
# kernel is timed after one warm-up call; the timings (seconds) go to stdout.
module Gds2ItvgKernel

export potential!

function potential!(out::Vector{Float64}, pts::Matrix{Float64}, p1::Matrix{Float64},
                    p2::Matrix{Float64}, w::Vector{Float64})
    n = size(pts, 2)
    ne = size(p1, 2)
    Threads.@threads :static for i in 1:n
        @inbounds begin
            x = pts[1, i]; y = pts[2, i]; z = pts[3, i]
            zs = abs(z)
            acc = 0.0
            for e in 1:ne
                x1 = x - p1[1, e]; y1 = y - p1[2, e]
                x2 = x - p2[1, e]; y2 = y - p2[2, e]
                r1 = sqrt(x1 * x1 + y1 * y1 + z * z)
                r2 = sqrt(x2 * x2 + y2 * y2 + z * z)
                num = z * (x1 * y2 - y1 * x2)
                den = zs * (r1 * r2 + x1 * x2 + y1 * y2 + zs * (zs + r1 + r2))
                acc += w[e] * atan(num, den)
            end
            out[i] = acc / pi
        end
    end
    return out
end

function main(args)
    inpath, outpath = args[1], args[2]
    repeats = length(args) >= 3 ? parse(Int, args[3]) : 0
    pts, p1, p2, w = open(inpath, "r") do io
        n = read(io, Int64); ne = read(io, Int64)
        pts = Matrix{Float64}(undef, 3, n); read!(io, pts)
        p1 = Matrix{Float64}(undef, 2, ne); read!(io, p1)
        p2 = Matrix{Float64}(undef, 2, ne); read!(io, p2)
        w = Vector{Float64}(undef, ne); read!(io, w)
        (pts, p1, p2, w)
    end
    out = zeros(size(pts, 2))
    potential!(out, pts, p1, p2, w)            # warm-up (includes JIT compilation)
    times = Float64[]
    for _ in 1:repeats
        push!(times, @elapsed potential!(out, pts, p1, p2, w))
    end
    open(outpath, "w") do io
        write(io, out)
    end
    repeats > 0 && println(join(times, " "))
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    Gds2ItvgKernel.main(ARGS)
end
