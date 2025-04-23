using Printf
using LinearAlgebra
import Roots: find_zero

export gmres!

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
"""
gmres!(x, A, b; m::Int=typemax(Int), rel_rtol::Real=1e-2, maxiter::Int=10, verbose::Bool=true, trace::Union{Nothing, GMRESTrace}=nothing) =
    _gmres_impl!(x, A, b, 0, m, false, trace, rel_rtol, maxiter, verbose)


"""
    gmres!(x, A, b, Δ; m::Int=typemax(Int), rel_rtol::Real=1e-2, maxiter::Int=10, verbose::Bool=true)

Using `x` as an initial guess, solve the problem 

    min_x `|A*x =b|` 
    s.t. |x| ≤ Δ

using the GMRES method and the hookstep method (see https://arxiv.org/pdf/0809.1498.pdf).
"""
gmres!(x, A, b, Δ::Real; m::Int=typemax(Int), rel_rtol::Real=1e-2, maxiter::Int=10, verbose::Bool=true, trace::Union{Nothing, GMRESTrace}=nothing) =
    _gmres_impl!(x, A, b, Δ, m, true, trace, rel_rtol, maxiter, verbose)

function _gmres_impl!(x, 
                      A,
                      b,
                      Δ::Real,
                      m::Int,
                      solve_hookstep::Bool,
                      trace::T,
                      rel_rtol::Real=1e-2,
                      maxiter::Int=10,
                      verbose::Bool=true,) where {T<:Union{Nothing, GMRESTrace}}
    # store norm of b
    b_norm = norm(b)

    # initial residual
    r  = similar(b)
    r .= b .- A*x

    # residual norm
    r_norm = norm(r)

    # setup Arnoldi iteration. This will modify `r`, so we compute its norm first 
    arnit = ArnoldiIteration(A, r)

    # right hand side
    g = Float64[r_norm]

    # start iterations
    it = 1; while true
        # run arnoldi step
        Q, H = arnoldi!(arnit)

        # grow right hand side
        push!(g, 0.0)

        # solve least squares problem
        # see Viswanath https://arxiv.org/pdf/0809.1498.pdf
        if solve_hookstep
            F = svd(H); U = F.U; d = F.S; V = F.V
            p = U'*g
            # find μ > 0 such that ||q|| = Δ
            fun(μ) = norm(p .* d ./ (μ.+ d.^2)) - Δ
            μ_0 = fun(0) > 0 ? find_zero(fun, 0) : zero(fun(0))
            # construct vector y that generates the hookstep
            y = V*(p .* d ./ (μ_0.+ d.^2))
        else
            y = arnit.H\g
        end

        # check convergence
        r_norm = norm(H*y - g)

        # print output with relative residual norm
        verbose && dispstatus(it, r_norm/b_norm)

        # reached convergence
        r_norm < rel_rtol*b_norm && (lincomb!(x, Q, y; add=true); break)
        it >= maxiter            && (lincomb!(x, Q, y; add=true); break)

        # restart
        if it % m == 0
            # get current estimate of solution
            lincomb!(x, Q, y; add=true)
            
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

    # update trace with final state
    if T <: GMRESTrace
        push!(trace, it, arnit.Q, arnit.H)
    end

    return x, r_norm, it
end

function dispstatus(it::Int, res)
    it == 1 && @printf "+-----------+------------+\n"
    it == 1 && @printf "| GMRES It. |  Res. norm | \n"
    it == 1 && @printf "+-----------+------------+\n"
               @printf "     %5d  | %6.5e\n" it res
    flush(stdout)
end