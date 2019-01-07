using Printf
using LinearAlgebra

export gmres!

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
    tol     : stop the iteration when `norm(A*xn-b)/norm(b) < tol` where
              `xn` is the current solution in the n-the krylov subspace.
    maxiter : stop the iteration after `maxiter` iterations.  
    verbose : whether to print iteration status
"""
function gmres!(A, b, rtol::Real=1e-2, maxiter::Int=10, verbose::Bool=true)
    # store norm of b
    bnorm = norm(b)

    # Set up arnoldi iteration. Note that `Q[1]` will shadow 
    # the same memory as `b`. Hence, when we perform 
    # `lincomb!(b, Q, y)` we might have issues if broadcasting
    # is not implemented correctly for `typeof(b)`
    arn = ArnoldiIteration(A, b)

    # right hand side
    g = Float64[bnorm]

    # residual norm
    res_norm = 1.0
    
    # start iterations
    it = 1; while true

        # run arnoldi step
        Q, H = arnoldi!(arn)

        # grow right hand side
        push!(g, 0.0)

        # solve least squares problem
        y = arn.H\g 

        # check convergence
        res_norm = norm(H*y - g)/bnorm

        # print output
        verbose && dispstatus(it, res_norm)

        # reached convergence
        res_norm < rtol && (lincomb!(b, Q, y); break)
        it >= maxiter   && (lincomb!(b, Q, y); break)

        # update
        it += 1
    end

    return b, res_norm
end

function dispstatus(it::Int, res)
    it == 1 && @printf "+-----------+------------+\n"
    it == 1 && @printf "| GMRES It. |  Res. norm | \n"
    it == 1 && @printf "+-----------+------------+\n"
               @printf "     %5d  | %6.5e\n" it res
    flush(stdout)
end

"""
    lincomb!(out, Q, y)

Compute linear combination of first `n` Arnoldi vectors in `Q` using weights in the 
vector `y` of length `n`, writing the result in `out`. With this function, the
solution in the full space is recovered from its projection `y` on the Krylov
subspace basis given by Arnoldi basis vectors `Q[1:n]`.
"""
function lincomb!(out::X, Q::Vector{X}, y::Vector{<:Real}) where X
    length(Q) == length(y)+1 || error("length(Q) must be length(y)+1")
    out .= Q[1].*y[1]
    for i = 2:length(y)
        out .+= Q[i].*y[i]
    end
    return out
end