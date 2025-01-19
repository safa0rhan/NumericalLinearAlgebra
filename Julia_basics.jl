# Julia Basics - A Quick Start Guide

# Defining varaibles
x = 10          # Integer variable
y = 3.14        # Float variable
name = "Julia"  # String variable

# Printing the variables
println("x = ", x)
println("Hello, Julia!")
@show x, y  # Prints values of x and y

# Defining a function
function f(x)
    return x^2
end

# Calling the function
println("f(10) = ", f(10))

# Defining a function with multiple arguments
function f(x, y)
    return x + y
end

# Calling the function
println("f(10, 20) = ", f(10, 20))

# Working with matrices
A = [1 2 3; 2 0 2; 3 2 9]  # Define a 3x3 matrix

# Printing the matrix
println("A = ", A)

# Accessing elements of the matrix
println("A[1, 1] = ", A[1, 1])

# Accessing and Modifying elements of the matrix
element = A[1, 3]    # Access element in row 1, column 3
A[1, 3] = 4          # Modify element at row 1, column 3

# Basic operations on matrices
B = [1 0 1; 0 1 0; 1 0 1]
C = A + B         # Matrix addition
D = A * B         # Matrix multiplication
E = A .+ 1        # Element-wise addition
transpose_A = A'  # Transpose of A

# Matrix-vector multiplication
x = [1, 0, -1]  # Column vector
y = A * x       # Matrix-vector multiplication

# Identity Matrix
I = Matrix{Float64}(I, 3, 3)  # Define a 3x3 identity matrix
println("I = ", I)
# Multiply a scalar with the identity matrix
scalar = 5.0
I_scaled = scalar * I
println("I_scaled = ", I_scaled)
# Eigenvalues and eigenvectors
# Solving linear equations
b = [1.0, 2.0, 3.0]  # Right-hand side vector
x = A \ b            # Solve the linear system Ax = b
println("Solution x = ", x)
using LinearAlgebra
# Rayleigh Quotient Iteration
function rayleigh_quotient_iteration(A, x0, max_iter)
    x = x0 / norm(x0)
    μ = (x' * A * x)[1]
    for i in 1:max_iter
        try
            x = (A - μ * I) \ x
        catch
            println("Matrix is singular, stopping iteration.")
            break
        end
        x = x / norm(x)
        μ = (x' * A * x)[1]
    end
    return μ, x
end

# Example usage
A = [4.0 1.0; 1.0 3.0]
x0 = [1.0, 0.0]
max_iter = 100
eigenvalue, eigenvector = rayleigh_quotient_iteration(A, x0, max_iter)
println("Eigenvalue: ", eigenvalue)
println("Eigenvector: ", eigenvector)
eigenvalues, eigenvectors = eigen(A)
λ = eigvals(A)    # Compute eigenvalues
λ, v = eigvecs(A) # Compute eigenvalues and eigenvectors

# Arnoldi factorization
H = [1 2 3; 0 0 0; 0 0 0]
V = [1 0 0; 0 1 0; 0 0 1]
r = [1, 2, 3]
k = 1
p = 1
H, V, r = arnoldi_sorensen(A, H, V, r, k, p)

# Working with verctors
x = [1.0, 0.0, -1.0]  # Column vector
# Operations on vectors
norm_x = norm(x)      # Compute vector norm
normalized_x = x / norm_x  # Normalize the vector
dot_product = dot(x, x)    # Dot product

# Random vectors and matrices
using Random
Random.seed!(42)  # Set seed for reproducibility
rand_vec = rand(3)  # Generate a random vector of size 3
rand_mat = rand(3, 3)  # Generate a random matrix of size 3x3

# Control Structures for Iterative Methods
for i in 1:100
    x = A * x
    x = x / norm(x)  # Normalize vector
end

# Conditional statements
if norm(x) < 1e-6
    println("Converged!")
else
    println("Not converged.")
end

# Error Analysis
λ_exact = eigvals(A)[1]      # Assume first eigenvalue is the exact solution
λ_approx = 1.2              # Example approximate eigenvalue
error = abs(λ_approx - λ_exact)  # Absolute error
log_error = log10(error)  # Logarithmic scale error

# Plotting
using Plots
x = range(0, 2π, length=100)
y = sin.(x)
plot(x, y, label="sin(x)", xlabel="x", ylabel="sin(x)", title="Plot of sin(x)")

# Plotting with Plots
using Pkg
Pkg.add("Plots")  # Install Plots package
using Plots        # Load Plots package

x = 1:10
y = rand(10)
plot(x, y, label="Random Values", xlabel="x-axis", ylabel="y-axis")

# Saving and Displaying Plots
savefig("plot.png")  # Save plot to file    
display("image/png", read("plot.png"))  # Display plot in Jupyter Notebook


semilogy(x, y, label="Semilog Plot")  # Semilog plot

# Defining and using a custom function
function power_method(A, x0, max_iter)
    for i in 1:max_iter
        x0 = A * x0
        x0 = x0 / norm(x0)  # Normalize
    end
    return x0
end

A = [2.0 1.0; 1.0 3.0]
x0 = [1.0, 0.0]
result = power_method(A, x0, 100)


# Performance Benchmarking
using BenchmarkTools
@btime A * x  # Measure execution time of matrix-vector multiplication

# Profiling
using Profile
@profile power_method(A, x0, 100)
Profile.print()

# Saving and Loading Data
using JLD
data = Dict("A" => A, "x0" => x0, "result" => result)
save("data.jld", data)  # Save data to file

loaded_data = load("data.jld")  # Load data from file
A_loaded = loaded_data["A"]
x0_loaded = loaded_data["x0"]
result_loaded = loaded_data["result"]

# Matrix factorization
using LinearAlgebra

# Singular Value Decomposition
U, Σ, V = svd(A)

# Cholesky Decomposition
A = [4.0 12.0 -16.0; 12.0 37.0 -43.0; -16.0 -43.0 98.0]
L = cholesky(A).L

# QR Decomposition
Q, R = qr(A)

# LU Decomposition
L, U = lu(A)

# QR Algorithm
A = [1.0 2.0 3.0; 2.0 0.0 2.0; 3.0 2.0 9.0]
λ = eigen(A).values
Q, R = qr(A)
for i in 1:100
    Q, R = qr(A)
    A = R * Q
end
λ_new = eigen(A).values

# Arnoldi-Sorensen Algorithm
function arnoldi_sorensen(A, H, V, r, k, p)
    tol = 1e-14
    for j in 1:p
        β = norm(r, 2)
        if β < tol
            println("Breakdown")
            return
        end
        ekjm1 = zeros(size(H, 2), 1)
        ekjm1[k + j - 1] = 1
        H = [H; β * ekjm1']
        v = r / β
        V = [V v]
        w = A * v
        h = V' * w
        H = [H h]
        r = w - V * h
        s = 1e60
        count = 0
        while norm(s) > eps() * norm(r)
            s = V' * r
            r = r - V * s
            h = h + s
            count = count + 1
            if count > 4
                println("Reorthogonalization failed")
            end
        end
    end
    return H, V, r
end
