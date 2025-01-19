using LinearAlgebra
using Printf
using BenchmarkTools

"""
    classical_gs_single(Q_sub, w, k)

Perform single-pass Classical Gram-Schmidt of vector `w` against 
the first k columns of `Q_sub`. 

Returns:
- h[1..k]: the coefficients from the projections
- beta   : the norm of the remainder
- worth  : the "raw" orthogonalized vector (before dividing by beta)
"""
function classical_gs_single(Q_sub, w, k)
    h = zeros(Float64, k)
    worth = copy(w)

    for j in 1:k
        h[j] = dot(Q_sub[:,j], worth)
        worth .-= h[j]*Q_sub[:,j]
    end
    beta = norm(worth)
    return h, beta, worth
end

"""
    modified_gs(Q_sub, w, k)

Perform single-pass Modified Gram-Schmidt of vector `w` 
against the first k columns of `Q_sub`.
"""
function modified_gs(Q_sub, w, k)
    worth = copy(w)
    h = zeros(Float64, k)
    for j in 1:k
        alpha = dot(Q_sub[:,j], worth)
        h[j] = alpha
        @inbounds worth .-= alpha*Q_sub[:,j]
    end
    beta = norm(worth)
    return h, beta, worth
end

"""
    classical_gs_double(Q_sub, w, k)

Double Classical Gram-Schmidt = single pass + second pass.
"""
function classical_gs_double(Q_sub, w, k)
    h, beta, worth = classical_gs_single(Q_sub, w, k)
    h2, beta2, worth2 = classical_gs_single(Q_sub, worth, k)
    h_total = h .+ h2
    beta_total = beta2
    return h_total, beta_total, worth2
end

"""
    classical_gs_triple(Q_sub, w, k)

Triple Classical Gram-Schmidt.
"""
function classical_gs_triple(Q_sub, w, k)
    h, beta, worth = classical_gs_single(Q_sub, w, k)
    h2, _, worth2   = classical_gs_single(Q_sub, worth, k)
    h3, beta3, worth3 = classical_gs_single(Q_sub, worth2, k)
    h_total = h .+ h2 .+ h3
    beta_total = beta3
    return h_total, beta_total, worth3
end

"""
    arnoldi_variant(A, b, m, gs_variant)

Mimics the logic of the provided "arnoldi.m" but in Julia, 
choosing among different Gram-Schmidt variants.

Returns (Q,H) where
- Q is size (n x (m+1)), 
- H is size ((m+1) x m).
"""
function arnoldi_variant(A, b, m, gs_variant::Symbol)
    n = length(b)
    Q = zeros(Float64, n, m+1)
    H = zeros(Float64, m+1, m)

    beta0 = norm(b)
    Q[:,1] = b / beta0

    for k in 1:m
        w = A * Q[:,k]

        if gs_variant == :classical1
            h, beta, worth = classical_gs_single(Q[:,1:k], w, k)
        elseif gs_variant == :modified
            h, beta, worth = modified_gs(Q[:,1:k], w, k)
        elseif gs_variant == :classical2
            h, beta, worth = classical_gs_double(Q[:,1:k], w, k)
        elseif gs_variant == :classical3
            h, beta, worth = classical_gs_triple(Q[:,1:k], w, k)
        else
            error("Unknown GS variant: $gs_variant")
        end

        H[1:k, k]   = h
        H[k+1,   k] = beta

        if beta < 1e-14
            break
        end

        Q[:,k+1] = worth / beta
    end

    return Q, H
end

function test_arnoldi_variants(A, b, m_vals)
    variants = [:classical1, :modified, :classical2, :classical3]
    results = Dict{Symbol,Dict{Symbol,Vector{Float64}}}()
    for v in variants
        results[v] = Dict(:time => Float64[], :orth => Float64[])
    end

    for m in m_vals
        println("\n===== m = $m =====")
        for v in variants
            t = @elapsed begin
                Q,H = arnoldi_variant(A, b, m, v)
            end
            Q,H = arnoldi_variant(A, b, m, v)
            actual_mplus1 = m+1
            orth_err = norm(Q[:,1:actual_mplus1]'*Q[:,1:actual_mplus1] - I(actual_mplus1))

            push!(results[v][:time], t)
            push!(results[v][:orth], orth_err)

            @printf("%-12s -> time = %.4g,  orth_error = %.3e\n",
                string(v), t, orth_err)
        end
    end
    return results
end

n = 300
A = randn(n,n)
b = randn(n)
m_vals = [5, 10, 20, 50, 100]

results = test_arnoldi_variants(A, b, m_vals)

println("\nRESULT SUMMARY:")
for (variant, data) in results
    println("\nVariant: $variant")
    for (j,mval) in enumerate(m_vals)
        t = data[:time][j]
        o = data[:orth][j]
        @printf("   m=%-3d  time=%8.3g   orth_err=%.3e\n", mval, t, o)
    end
end
