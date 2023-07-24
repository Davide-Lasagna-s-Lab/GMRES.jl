export eigvals, eigs

function eigs(A, v, nev::Int, niter::Int)

    # setup arnoldi iteration
    arnit = ArnoldiIteration(A, v)

    for iter = 1:(niter-1)
        Q, H = arnoldi!(arnit)
    end

    # run last step
    Q, H = arnoldi!(arnit)

    # perform decomposition
    out = eigen!(H[1:end-1, :])

    # get sorting permutation by magnitude
    idxs = sortperm(out.values; by=abs, rev=true)

    return (out.values[idxs[1:nev]], 
            [lincomb!(similar(v), Q, out.vectors[:, idx]) for idx in idxs[1:nev]])
end

function eigvals(A, v, nev::Int, niter::Int)

    # setup arnoldi iteration
    arnit = ArnoldiIteration(A, v)

    # vals
    evals = Vector{Complex{Float64}}[]

    for iter = 1:niter

        # run arnoldi step
        Q, H = arnoldi!(arnit)

        # get eigenvalues of H and sort them in descending order by magnitude part
        vals = sort(LinearAlgebra.eigvals(H[1:end-1, :]); by=abs, rev=true)

        # add one vector to evals if needed, padding the initial part with zeros
        if length(evals) < nev
            push!(evals, Complex{Float64}[0 for i = 1:iter-1])
        end

        # push eigenvalues to history
        for i = eachindex(evals)
            push!(evals[i], vals[i])
        end

        # check all eigenvalues have variation less than tol in the last iteration
        # if maximum(abs(evals[end-1] - evals[end]) for i = 1:length(evals)) < tol
        #    break
        # end
    end

    return evals
end