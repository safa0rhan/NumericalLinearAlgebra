using LinearAlgebra

"""
    build_K(A, b, m)

Construct the n x m matrix K_m = [b, A*b/||A*b||, ..., A^(m-1)*b/||A^(m-1)*b||].
"""
function build_K(A, b, m)
    n = size(A,1)
    K = zeros(eltype(A), n, m)
    v = copy(b)
    normv = norm(v)
    K[:,1] = v / normv
    for j in 2:m
        v = A * v
        nv = norm(v)
        K[:,j] = v / nv
    end
    return K
end

"""
    approximate_eigs_primitive(A, b, m)

Compute the eigenvalues of (K_m^T K_m)^{-1} * (K_m^T A K_m),
where K_m is the "primitive" basis (Power Method iterates).
"""
function approximate_eigs_primitive(A, b, m)
    K = build_K(A, b, m)
    M = K' * K             # size m x m
    N = K' * A * K         # size m x m
    mat = inv(M) * N
    return eigvals(mat)    # an array of length m
end


"""
    arnoldi_doubleGS(A, b, m)

Perform m-step Arnoldi on A with double Classical Gram-Schmidt.
Return (Q, H) with:
  - Q: (n x (m+1)) basis
  - H: ((m+1) x m) upper-Hessenberg
"""
function arnoldi_doubleGS(A, b, m)
    n = size(A,1)
    Q = zeros(eltype(A), n, m+1)
    H = zeros(eltype(A), m+1, m)

    # init
    nb = norm(b)
    Q[:,1] = b / nb

    for k in 1:m
        w = A * Q[:,k]

        # 1st pass
        h1 = Vector{eltype(A)}(undef, k)
        for j in 1:k
            h1[j] = dot(Q[:,j], w)
            @inbounds w .-= h1[j] * Q[:,j]
        end

        # 2nd pass
        h2 = Vector{eltype(A)}(undef, k)
        for j in 1:k
            alpha = dot(Q[:,j], w)
            h2[j] = alpha
            @inbounds w .-= alpha * Q[:,j]
        end

        # sum the two
        for j in 1:k
            H[j,k] = h1[j] + h2[j]
        end

        H[k+1,k] = norm(w)
        if H[k+1,k] < 1e-14
            # breakdown or subspace complete
            break
        end
        Q[:,k+1] = w / H[k+1,k]
    end

    return Q,H
end

"""
    arnoldi_ritz_values(A, b, m)

Run m-step Arnoldi with double GS, 
return the eigenvalues (Ritz values) of the leading m x m block of H.
"""
function arnoldi_ritz_values(A, b, m)
    Q,H = arnoldi_doubleGS(A, b, m)
    Hsub = H[1:m, 1:m]
    return eigvals(Hsub)
end




using Plots

# Suppose we have a matrix from the previous exercise:
# e.g. A = [some 100x100 or 3x3 or 200x200...]
# For demonstration, let's pick a random 5x5 matrix:
A_demo = randn(5,5)
b_demo = randn(5)

m_max = 5

# We'll store the approximate eigenvalues in arrays/dicts
# For each m, we'll store the "primitive" eigenvals and the "Arnoldi" Ritz eigenvals.
approx_primitive = Dict{Int,Vector{ComplexF64}}()
approx_arnoldi    = Dict{Int,Vector{ComplexF64}}()

for m in 1:m_max
    # Primitive approach
    λp = approximate_eigs_primitive(A_demo, b_demo, m)
    # Arnoldi approach
    λa = arnoldi_ritz_values(A_demo, b_demo, m)

    approx_primitive[m] = λp
    approx_arnoldi[m]   = λa

    println("m = $m")
    println("  Primitive eigenvals: ", λp)
    println("  Arnoldi   eigenvals: ", λa)
    println()
end

# You might plot the real parts vs m, or just print them.  
# For a larger matrix, you could also compare the error to the true eigenvalues
# if you can compute them (with eig(A_demo)).
