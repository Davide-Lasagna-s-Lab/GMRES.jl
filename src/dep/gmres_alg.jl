using Printf
using LinearAlgebra
import Roots: find_zero

"""
    gmres!(A, b, rtol=1e-2, maxiter=10, verbose=True)

Solve `Ax=b` using the GMRES method. The input `b` is overwritten during 
the solution, and it is returned at completion of the algorithm. The inputs
`A` and `b` are not typed, and can be anything as long as they satisfy the 
following interface:
1) `b` must be broadcast-able
2) `b` must support `norm(b)` and `dot(b, b)`
3) `A`, `b` must support the product `A*b`
Note that in 3) we do not check that `size` of `A` and `b` match, but 
these must be somehow conforming.

The keyword arguments are used to control the iterations:
    rtol     : stop the iteration when `norm(A*xn-b)/norm(b) < rtol` where
              `xn` is the current solution in the n-the krylov subspace.
    maxiter : stop the iteration after `maxiter` iterations.  
    verbose : whether to print iteration status
    m       : restart GMRES every m iterations
    Δ       : trust region radius
"""
gmres!(A, b; m::Int=typemax(Int), rtol::Real=1e-2, maxiter::Int=10, verbose::Bool=true) =
    (Base.depwarn("This interface for gmres! is deprecated", :gmres!, force=true); _gmres_impl!(A, b, 0, m, false, rtol, maxiter, verbose))

gmres!(A, b, Δ::Real; m::Int=typemax(Int), rtol::Real=1e-2, maxiter::Int=10, verbose::Bool=true) =
    (Base.depwarn("This interface for gmres! is deprecated", :gmres!, force=true); _gmres_impl!(A, b, Δ, m, true, rtol, maxiter, verbose))

function _gmres_impl!(A,
                      b,
                      Δ::Real,
                      m::Int,
                      solve_hookstep::Bool,
                      rtol::Real=1e-2,
                      maxiter::Int=10,
                      verbose::Bool=true)
    # store norm of b
    bnorm = norm(b)

    # right hand side
    g = Float64[bnorm]

    arnit = ArnoldiIteration(A, b)

    # residual norm
    res_norm = 1.0
    
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
            fun(μ) = norm(p .* d ./ (μ .+ d.^2)) - Δ
            μ_0 = fun(0) > 0 ? find_zero(fun, 0) : zero(fun(0))
            # construct vector y that generates the hookstep
            y = V*(p .* d ./ (μ_0 .+ d.^2))
        else
            y = arnit.H\g
        end

        # check convergence
        res_norm = norm(H*y - g)/bnorm

        # print output
        verbose && dispstatus(it, res_norm)

        # reached convergence
        res_norm < rtol && (lincomb!(b, Q, y); break)
        it >= maxiter   && (lincomb!(b, Q, y); break)

        # restart
        if it % m == 0 
            # right hand side
            g = Float64[bnorm]

            # get current estimate of b
            lincomb!(b, Q, y)
            
            # restart iteration
            arnit = ArnoldiIteration(A, b)
        end
        
        # update
        it += 1
    end

    return b, res_norm, it
end
