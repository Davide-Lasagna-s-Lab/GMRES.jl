export ArnoldiIteration, arnoldi!

# INTERFACE
# b must be broadcast-able
# b must support norm(b), dot(b, b)
# A, b must support A*b
# we do not check the `size` of A and b

mutable struct ArnoldiIteration{X, Op}
    A::Op
    Q::Vector{X}
    H::Matrix{Float64}
    n::Int
    function ArnoldiIteration{X, Op}(A::Op, b::X) where {X, Op}
        b ./= norm(b)
        Q = X[]
        push!(Q, b)
        new(A, Q, zeros(0, 0), 1)
    end
end

ArnoldiIteration(A, b) = 
    ArnoldiIteration{typeof(b), typeof(A)}(A, b)

# Run arnoldi step
function arnoldi!(arn::ArnoldiIteration)
    # aliases
    A, Q, n = arn.A, arn.Q, arn.n
    
    # allocate new matrix H
    H = zeros(n+1, n)
    
    # copy last iteration
    H[1:n, 1:n-1] = arn.H 

    # store
    arn.H = H

    # create new vector by successive orthogonalisation
    v = A*Q[n]
    for j = 1:n
        q = Q[j]
        h = dot(v, q)
        v .= v .- h.*q
        H[j, n] = h
    end
    H[n+1, n] = norm(v)
    v ./= H[n+1, n]
    push!(Q, v)

    # update
    arn.n += 1
    
    return Q, H
end