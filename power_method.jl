#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Plots

"""
    power_method(A, x0; maxit=100, tol=1e-12)

Power method for dominant (largest magnitude) eigenvalue of matrix A.
Returns two arrays: 
- `λvals`: approximate eigenvalue at each iteration
- `resids`: error measure or residual at each iteration (optional)

We measure the approximate eigenvalue via the Rayleigh quotient in each iteration.
"""
function power_method(A, x0; maxit=100, tol=1e-12)
    x = copy(x0)
    λvals = Float64[]
    for k in 1:maxit
        # Power iteration step
        y = A*x
        # Normalization
        x = y / norm(y)

        # Approx eigenvalue from Rayleigh quotient
        λ_approx = (x' * A * x) / (x' * x)
        push!(λvals, λ_approx)
        
        # Optional stopping criterion
        if k > 1
            if abs(λvals[end] - λvals[end-1]) < tol
                break
            end
        end
    end
    return λvals
end

"""
Helper function to get exact eigenvalues of a 3x3 matrix for reference.
"""
function exact_eigs_3x3(A)
    return eigvals(A)  # returns eigenvalues (not necessarily sorted)
end

# Example usage for part (a):
A = [1 2 3;
     2 0 2;
     3 2 9]
x0 = [1.0, 0.0, -1.0]

# Compute exact eigenvalues for reference
λs = exact_eigs_3x3(A)
# Let's define largest eigenvalue in magnitude:
λ1 = λs[argmax(abs.(λs))]

# Run power method
λvals_pm = power_method(A, x0, maxit=50)

# Compute error for plotting
errors_pm = abs.(λvals_pm .- λ1)

# Semilog plot of error
p_pm = plot(errors_pm, yaxis=:log, 
    title="Power Method: Error in largest eigenvalue vs iteration",
    xlabel="Iteration", ylabel="|λ1 - λ^(k)| (log scale)",
    markershape=:circle, legend=false)
display(p_pm)
