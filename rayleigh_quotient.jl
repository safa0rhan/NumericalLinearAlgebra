"""
    rayleigh_quotient_iteration(A, x0; maxit=100, tol=1e-12)

Rayleigh quotient iteration for a single eigenvalue.
Returns an array `λvals` of approximate eigenvalues at each iteration.
"""
function rayleigh_quotient_iteration(A, x0; maxit=100, tol=1e-12)
    x = copy(x0)
    x ./= norm(x)  # normalize
    λvals = Float64[]

    for k in 1:maxit
        # Rayleigh quotient
        λ_approx = (x' * A * x) / (x' * x)
        push!(λvals, λ_approx)

        # Solve (A - λ_approx I)*y = x
        y = (A - λ_approx*I) \ x
        x = y / norm(y)

        # Optional stopping criterion
        if k > 1
            if abs(λvals[end] - λvals[end-1]) < tol
                break
            end
        end
    end
    return λvals
end

A = [1 2 3;
     2 0 2;
     3 2 9]
x0 = [1.0, 0.0, -1.0]

# Part (b): Run RQI on original A
λvals_rqi_b = rayleigh_quotient_iteration(A, x0, maxit=50)
errors_rqi_b = abs.(λvals_rqi_b .- λ1)

p_rqi_b = plot(errors_rqi_b, yaxis=:log,
    title="Rayleigh Quotient Iteration (original A)",
    xlabel="Iteration", ylabel="|λ1 - λ^(k)| (log scale)",
    markershape=:circle, legend=false)
display(p_rqi_b)



A_mod = [1 2 4;
         2 0 2;
         3 2 9]

λs_mod = eigvals(A_mod)
λ1_mod = λs_mod[argmax(abs.(λs_mod))]

λvals_rqi_c = rayleigh_quotient_iteration(A_mod, x0, maxit=50)
errors_rqi_c = abs.(λvals_rqi_c .- λ1_mod)

p_rqi_c = plot(errors_rqi_c, yaxis=:log,
    title="Rayleigh Quotient Iteration (modified A_{1,3} = 4)",
    xlabel="Iteration", ylabel="|λ1 - λ^(k)| (log scale)",
    markershape=:circle, legend=false)
display(p_rqi_c)
