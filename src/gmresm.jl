using Printf
using LinearAlgebra
import Roots: find_zero

export gmres!, hookstep!

"""
    gmres!(x, A, b; m::Int=typemax(Int), rel_rtol::Real=1e-2, maxiter::Int=10, verbose::Bool=true)

Solve the linear system `A*x = b` using the GMRES method. The argument `x` is used
as an initial guess and is overwritten with the solution upon return. 

The inputs `A` and `b` are not typed, and can be anything as long as they satisfy the 
following interface:
1) `b` must be broadcast-able
2) `b` must support `norm(b)` and `dot(b, b)`
3) `A`, `b` must support the product `A*b`
Note that in 3) we do not check that `size` of `A` and `b` match, but 
these must be somehow conforming.

The keyword arguments are used to control the iterations:
    rel_rtol : stop the iteration when `norm(A*xn-b)/norm(b) < rel_rtol` where
               `xn` is the current solution in the n-the Krylov subspace.
    maxiter  : stop the iteration after `maxiter` iterations.  
    verbose  : whether to print iteration status
    m        : restart GMRES every m iterations
    callback : optional callback function to keep track of values during operation,
               with a signature `callback(it::Int, r_norm::Float64, Q::Vector{typeof(x)}, H::Matrix{Float64}, y::Vector{Float64})`
"""
gmres!(x, A, b; m::Int=typemax(Int), rel_rtol::Real=1e-2, maxiter::Int=10, verbose::Bool=true, callback=(args...)->false) =
    _gmres_impl!(x, A, b, 0, m, false, rel_rtol, maxiter, verbose, callback)


"""
    gmres!(x, A, b, Δ; m::Int=typemax(Int), rel_rtol::Real=1e-2, maxiter::Int=10, verbose::Bool=true) -> (x, r_norm, it, K)

Using `x` as an initial guess, solve the problem

    min_x `|A*x =b|`
    s.t. |x| ≤ Δ

using the GMRES method and the hookstep method (see https://arxiv.org/pdf/0809.1498.pdf).

As in that method, the iteration stops when the residual of the unconstrained least-squares solution in the Krylov
space is below `rel_rtol*norm(b)` (or after `maxiter` iterations), and the solution returned is the hookstep in that
Krylov space: the minimiser of `|A*x - b|` with `|x - x₀| ≤ Δ`, `x₀` the initial guess. When `Δ` binds, the residual
of the hookstep has a floor above the tolerance that more Krylov vectors barely lower. `r_norm` is the residual of
the hookstep. `K` is the Krylov space, which [`hookstep!`](@ref) uses to compute the hookstep for another radius
without new products with `A`. The callback receives the hookstep and its residual at each iteration.
"""
gmres!(x, A, b, Δ::Real; m::Int=typemax(Int), rel_rtol::Real=1e-2, maxiter::Int=10, verbose::Bool=true, callback=(args...)->false) =
    _gmres_impl!(x, A, b, Δ, m, true, rel_rtol, maxiter, verbose, callback)

"""
    hookstep!(x, K, Δ) -> (x, r_norm)

The hookstep for the radius `Δ` in the Krylov space `K` returned by `gmres!(x, A, b, Δ₀)`, with no product with `A`:
`x` becomes the initial guess of that call plus the minimiser of `|A*x - b|` over the Krylov space with
`|x - x₀| ≤ Δ`, and `r_norm` is its residual. A trust-region method calls it after a rejected step, when the
operator and the right-hand side have not changed.
"""
function hookstep!(x, K, Δ::Real)
    y, _ = _hookstep(K.H, K.g, Δ)
    x .= K.x₀
    lincomb!(x, K.Q, y; add=true)
    return x, norm(K.H*y - K.g)
end

# The hookstep of Viswanath in the Krylov space: the y that minimises ‖H y - g‖ with ‖y‖ ≤ Δ, from the SVD of H;
# and the residual of the unconstrained least-squares solution, ‖(I - U Uᵀ) g‖, with no division by the singular
# values
function _hookstep(H, g, Δ)
    F = svd(H); U = F.U; d = F.S; V = F.V
    p = U'*g

    # find μ > 0 such that ||q|| = Δ
    fun(μ) = norm(p .* d ./ (μ.+ d.^2)) - Δ
    μ_0 = fun(0) > 0 ? find_zero(fun, 0) : zero(fun(0))

    # construct vector y that generates the hookstep
    y = V*(p .* d ./ (μ_0.+ d.^2))

    return y, norm(U*p - g)
end

function _gmres_impl!(x, 
                      A,
                      b,
                      Δ::Real,
                      m::Int,
                      solve_hookstep::Bool,
                      rel_rtol::Real=1e-2,
                      maxiter::Int=10,
                      verbose::Bool=true,
                      callback=(args...)->false)
    # store norm of b
    b_norm = norm(b)

    # the initial guess, for the hookstep for other radii
    x₀ = copy(x)

    # initial residual (no product with A for a zero initial guess)
    r  = similar(b)
    norm(x) == 0 ? (r .= b) : (r .= b .- A*x)

    # residual norm
    r_norm = norm(r)

    # setup Arnoldi iteration. This will modify `r`, so we compute its norm first 
    arnit = ArnoldiIteration(A, r)

    # right hand side
    g = Float64[r_norm]

    # update callback
    callback(0, r_norm, typeof(x)[], Matrix{Float64}(undef, 0, 0), Float64[]) &&
        return solve_hookstep ? (x, r_norm, 0, nothing) : (x, r_norm, 0)

    # start iterations
    it = 1; while true
        # run arnoldi step
        Q, H = arnoldi!(arnit)

        # grow right hand side
        push!(g, 0.0)

        # solve least squares problem
        # see Viswanath https://arxiv.org/pdf/0809.1498.pdf
        if solve_hookstep
            y, r_newton = _hookstep(H, g, Δ)
        else
            y = arnit.H\g
        end

        # check convergence
        r_norm = norm(H*y - g)

        # print output with relative residual norm
        verbose && dispstatus(it, r_norm/b_norm)

        # update callback
        callback(it, r_norm, Q, H, y) && break

        # reached convergence: for the hookstep, that of the unconstrained solution in the Krylov space
        r_test = solve_hookstep ? r_newton : r_norm
        r_test < rel_rtol*b_norm && (lincomb!(x, Q, y; add=true); break)
        it >= maxiter            && (lincomb!(x, Q, y; add=true); break)

        # restart
        if it % m == 0
            # get current estimate of solution, the origin of the next Krylov space
            lincomb!(x, Q, y; add=true)
            x₀ .= x

            # get new residual
            r .= b .- A*x

            # residual norm
            r_norm = norm(r)

            # right hand side
            g = Float64[r_norm]

            # restart iteration
            arnit = ArnoldiIteration(A, r)
        end
        
        # update
        it += 1
    end

    # the hookstep also returns its Krylov space (that of the last restart), for hookstep!
    solve_hookstep && return x, r_norm, it, (Q=arnit.Q, H=arnit.H, g=g, x₀=x₀)
    return x, r_norm, it
end

function dispstatus(it::Int, res)
    it == 1 && @printf "+-----------+------------+\n"
    it == 1 && @printf "| GMRES It. |  Res. norm | \n"
    it == 1 && @printf "+-----------+------------+\n"
               @printf "     %5d  | %6.5e\n" it res
    flush(stdout)
end
