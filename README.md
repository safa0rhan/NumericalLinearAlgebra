# Numerical Linear Algebra

## Overview
This repository contains implementations of key numerical methods related to matrix computations. These methods were developed as part of the **SF2524 Matrix Computations for Large-scale Systems** course at KTH Royal Institute of Technology.

## Implemented Methods
The repository includes the following Julia implementations:
- **Arnoldi Iteration** (`arnoldi.jl`, `arnoldi2.jl`, `direct_arnoldi.jl`): Algorithms for generating Krylov subspaces and approximating eigenvalues of large matrices.
- **Power Method** (`power_method.jl`): A simple iterative algorithm for finding the dominant eigenvalue and eigenvector of a matrix.
- **Rayleigh Quotient Iteration** (`rayleigh_quotient.jl`): An advanced method for eigenvalue computation that converges rapidly near an eigenvalue.
- **Bwedgemat** (`Bwedge.mat`): A sample matrix used for testing and demonstrating the algorithms.

## How to Use
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/safa0rhan/NumericalLinearAlgebra.git
   cd NumericalLinearAlgebra
   ```

2. **Run Julia Scripts**:
   Ensure Julia is installed on your system. Then, execute any of the scripts:
   ```bash
   julia arnoldi.jl
   ```

## Prerequisites
- **Julia**: Ensure that Julia is installed. You can download it from [Julia's official website](https://julialang.org/downloads/).
- Any additional dependencies are specified in the scripts and can be added using Julia's package manager.
