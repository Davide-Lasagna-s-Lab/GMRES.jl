# Using the Arnoldi iteration, approximate the spectrum of the
# operator `A` as the Ritz values of the upper-Hessenberg
# matrix `H`

"""
    arnoldi_eigvals(A, b[; m=length(b), ϵ=1e-6])

Approximate the eigenvalues of `A` via Arnoldi iteration, starting from
the vector `b`.

`A` need not be formed explicitly - any object or function supporting
the action `A*x` (a dense/sparse matrix, a `LinearMap`, or a plain
function `x -> A*x`) can be passed, making this suitable for matrix-free
operators accessible only through their action on vectors.

# Arguments
- `A`: the operator whose eigenvalues are sought, applied only via
  `A*x` (or `A(x)` for a function/callable).
- `b`: starting vector. Its length sets the operator dimension `n`, and
  its direction sets the initial Krylov vector; it should have a
  nonzero component along any eigenvector you want the iteration to
  resolve.

# Keyword arguments
- `m::Int = length(b)`: maximum number of Arnoldi steps to run. The
  default allows the Krylov subspace to span all of `ℝⁿ`/`ℂⁿ` in
  exact arithmetic, at which point (barring rounding error) every Ritz
  value should match an eigenvalue of `A`.
- `ϵ::Real=1e-6`: breakdown tolerance. If the norm of the newly
  orthogonalized Krylov vector falls below `ϵ` (relative to the current
  Hessenberg column) at some step `k < m`, the Krylov subspace is
  treated as (numerically) `A`-invariant and the iteration stops early;
  the Ritz values of the resulting `k×k` Hessenberg matrix are then
  essentially exact eigenvalues of `A`, not just approximations.

# Returns
- A vector of Ritz values approximating eigenvalues of `A`. Its length
  equals the number of Arnoldi steps actually taken (`m`, unless
  breakdown occurred first).
- A vector of Ritz residual values approximating the error in the eigenvalue
  approximation.

# Notes
No restarting or extra re-orthogonalization is performed beyond what the
underlying Arnoldi step does, so accuracy (especially for non-normal
`A`) can degrade as `m` grows and orthogonality is lost. Prefer
checking the per-eigenvalue residual `‖A x_i - λ_i x_i‖` (e.g. via
`ritz_residuals`) over trusting a Ritz value from this function alone,
particularly for interior eigenvalues or non-normal operators.

# Examples
```julia
julia> A = randn(50, 50);

julia> b = randn(50);

julia> λ = arnoldi_eigvals(A, b; m=20);

julia> length(λ)
20
```
"""
function arnoldi_eigvals(A, b; m=length(b), ϵ=1e-6)
    arnit = ArnoldiIteration(A, b)
    local Λ, res
    for i in 1:m
        _, H = arnoldi!(arnit)
        k = size(H, 2)
        F = eigen(@view(H[1:k, 1:k]))
        Λ = F.values
        # https://www.netlib.org/utk/people/JackDongarra/etemplates/node216.html
        res = [H[i + 1, i]*abs(F.vectors[end, j]) for j in 1:i]

        norm(res) < ϵ && break
    end
    return Λ, res
end
