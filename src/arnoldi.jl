export ArnoldiIteration, arnoldi!

# INTERFACE
# b must be broadcast-able
# b must support norm(b), dot(b, b)
# A, b must support A*b
# we do not check the `size` of A and b

# ! This implementation seems a little inefficient, with what appears to unnecessary allocations each call to arnoldi!
mutable struct ArnoldiIteration{X, Op}
    A::Op              # The linear operator
    Q::Vector{X}       # Set of Arnoldi vectors
    H::Matrix{Float64} # The upper Hessemberg matrix
    function ArnoldiIteration{X, Op}(A::Op, b::X) where {X, Op}
        b ./= norm(b)
        Q = X[]
        push!(Q, b)
        new(A, Q, zeros(0, 0))
    end
end

ArnoldiIteration(A, b) = 
    ArnoldiIteration{typeof(b), typeof(A)}(A, b)

# Run arnoldi step
function arnoldi!(arn::ArnoldiIteration)
    # aliases
    A, Q, n = arn.A, arn.Q, length(arn.Q)

    # allocate new matrix H
    H = zeros(n+1, n)
    
    # copy last iteration
    H[1:n, 1:n-1] = arn.H

    # store
    arn.H = H

    # classical Gram-Schmid procedure with refinement
    # see http://slepc.upv.es/documentation/reports/str1.pdf
    v = A*Q[n]
    # @show norm(v[1]) norm(v[2]) norm(v[3]) norm(v[4]) norm(v[5]) norm(v[6]) norm(v[7]) norm(v[8])
    # @show norm(Q[1][1]) norm(Q[1][2]) norm(Q[1][3]) norm(Q[1][4]) norm(Q[1][5]) norm(Q[1][6]) norm(Q[1][7]) norm(Q[1][8])

    for _ = 1:2
        for j = 1:n
            h = dot(v, Q[j])
            v .-= h.*Q[j]
            H[j, n] += h
        end
    end

    H[n+1, n] = norm(v)
    v ./= H[n+1, n]
    push!(Q, v)

    return Q, H
end


"""
    lincomb!(out, Q, y; add=false)

Compute linear combination of first `n` Arnoldi vectors in `Q` using weights in the 
vector `y` of length `n`, writing the result in `out`. With this function, the
solution in the full space is recovered from its projection `y` on the Krylov
subspace basis given by Arnoldi basis vectors `Q[1:n]`.

If the keyword argument `add` is true, the linear combination of the columns of
`Q` is added to out, rather than overwriting its content.
"""
function lincomb!(out::X, Q::Vector{X}, y::Vector; add::Bool=false) where X
    length(Q) == length(y)+1 || error("length(Q) must be length(y)+1")
    if add
        out .+= Q[1].*y[1]
    else
        out .= Q[1].*y[1]
    end
    for i = 2:length(y)
        out .+= Q[i].*y[i]
    end
    return out
end