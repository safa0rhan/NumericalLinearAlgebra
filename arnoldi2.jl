using LinearAlgebra
using BenchmarkTools
using MAT
using Plots
using Random
using Printf

"""
    arnoldi(A, b, m, gs_variant)

Compute m Arnoldi iterations with different Gram-Schmidt variants.

Inputs:
- `A`          : matrix or linear operator
- `b`          : initial vector
- `m`          : number of Arnoldi steps
- `gs_variant` : symbol, one of `:classical1`, `:modified`, `:classical2`, `:classical3`

Outputs:
- `Q` in ℝ^(n x m)  (orthonormal columns)
- `H` in ℝ^(m+1 x m)
"""
function arnoldi(A, b, m, gs_variant::Symbol)
    n = size(A,1)
    Q = zeros(Float64, n, m)
    H = zeros(Float64, m+1, m)
    # Initialize
    q1 = b / norm(b)
    Q[:,1] = q1

    for k in 1:m
        # Arnoldi step
        v = A*Q[:,k]

        if gs_variant == :classical1
            # Single classical Gram-Schmidt
            v, H[1:k,k] = classical_gs_single(Q[:,1:k], v)

        elseif gs_variant == :modified
            # Modified GS
            v, H[1:k,k] = modified_gs(Q[:,1:k], v)

        elseif gs_variant == :classical2
            # Double classical GS
            v, H[1:k,k] = classical_gs_single(Q[:,1:k], v)
            # second pass
            v2, delta = classical_gs_single(Q[:,1:k], v)
            H[1:k,k] .+= delta
            v = v2

        elseif gs_variant == :classical3
            # Triple classical GS
            v, H[1:k,k] = classical_gs_single(Q[:,1:k], v)
            for _ in 1:2
                v2, delta = classical_gs_single(Q[:,1:k], v)
                H[1:k,k] .+= delta
                v = v2
            end
        else
            error("Unknown GS variant")
        end

        h_next = norm(v)
        if k < m
            H[k+1,k] = h_next
            if abs(h_next) > 1e-14
                Q[:,k+1] = v / h_next
            else
                # We found a Krylov subspace dimension smaller than m
                # no need to continue
                return Q,H
            end
        end
    end

    return Q,H
end

"""
    classical_gs_single(Q_sub, v)

Single-pass Classical Gram-Schmidt:
Orthogonalize vector v against the columns in Q_sub.

Returns:
- `v_new`: the updated (partially) orthogonal v
- `coeffs`: the vector of inner products (h_ij in Arnoldi)
"""
function classical_gs_single(Q_sub, v)
    coeffs = Q_sub' * v
    v_new = v - Q_sub*coeffs
    return v_new, coeffs
end

"""
    modified_gs(Q_sub, v)

Single-pass Modified Gram-Schmidt.
"""
function modified_gs(Q_sub, v)
    m = size(Q_sub,2)
    coeffs = zeros(Float64, m)
    v_new = copy(v)
    for j in 1:m
        alpha = dot(Q_sub[:,j], v_new)
        coeffs[j] = alpha
        v_new .-= alpha * Q_sub[:,j]
    end
    return v_new, coeffs
end


function test_arnoldi_variants(A, b, m_vals)
    gs_variants = [:classical1, :modified, :classical2, :classical3]
    results = Dict{Symbol,Any}()

    for v in gs_variants
        results[v] = (time=[], orth=[])
    end

    for m in m_vals
        @printf("\n=== m = %d ===\n", m)
        for v in gs_variants
            t = @elapsed begin
                Q,H = arnoldi(A, b, m, v)
            end
            actual_m = size(Q,2)
            orth_err = norm(Q' * Q - I(actual_m))
            
            push!(results[v][:time], t)
            push!(results[v][:orth], orth_err)

            @printf("%-12s: time = %8.4g, orth = %.4e\n", string(v), t, orth_err)
        end
    end
    return results
end

n = 200
A_big = randn(n,n);
b_big = randn(n)

m_values = [5, 10, 20, 50, 100]
results = test_arnoldi_variants(A_big, b_big, m_values)

println("\nSummary of results:\n")
for v in keys(results)
    println("Variant: $v")
    println("m | time  | orth")
    for (idx, m) in enumerate(m_values)
        t_ = results[v][:time][idx]
        o_ = results[v][:orth][idx]
        println("m=$m : time = $(round(t_, digits=4)), orth = $(round(o_, digits=4))")
    end
end
