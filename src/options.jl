using Parameters

export GMRESOptions

# ~~~ SEARCH OPTIONS FOR GMRES ITERATIONS ~~~

@with_kw struct GMRESOptions
    maxiter::Int  = 100    # maximum number of iterations
    verbose::Bool = true   # print iteration status or stay silent
    tol::Float64  = 1e-10  # relative tolerance for convergence
    @assert maxiter > 0    # bad behaviour occurs if maxiter is 0
end