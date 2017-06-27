export gmres!

# Data structure for convergence analysis.
mutable struct ConvergenceResults
    hist::Vector{Float64}
    status::Symbol
end
ConvergenceResults() = ConvergenceResults(Float64[1.0], :unknown)

Base.push!(convres::ConvergenceResults, r::Real) = push!(convres.hist, r)

"""
    gmres!(A, b; [tol=1e-6], [maxiter=10])

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
"""
function gmres!(A, b, opts::GMRESOptions=GMRESOptions())
    # store norm of b
    bnorm = norm(b)

    # store trace
    convres = ConvergenceResults()

    # set up arnoldi iteration
    arn = ArnoldiIteration(A, b)

    # right hand side
    g = Float64[bnorm]
    
    # start iterations
    it = 1; while true

        # run arnoldi step
        Q, H = arnoldi!(arn)

        # grow right hand side
        push!(g, 0.0)

        # solve least squares problem
        y = arn.H\g 

        # check convergence
        rnorm = norm(H*y - g)

        # store trace
        push!(convres, rnorm/bnorm)

        # reached tolerance
        if rnorm < opts.tol
            # update convergence status
            convres.status = :converged
            lincomb!(b, Q, y); break
        end

        # reached max iterations
        if it >= opts.maxiter
            convres.status = :maxiterreached
            lincomb!(b, Q, y); break
        end

        # update
        it += 1    
    end

    return b, convres
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