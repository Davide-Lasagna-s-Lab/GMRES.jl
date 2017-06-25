using GMRES
using Base.Test

@testset "symmetric     " begin
    # construct random symmetric matrix and rhs
    A = randn(400, 400)
    A = A + A'
    b = randn(400)

    # create iterator
    arn = ArnoldiIteration(A, b)

    # the square part must be symmetric to machine accuracy
    for i = 1:6
        Q, H = arnoldi!(arn)
        display(H); println()
        println(length(Q))
        # @test issymmetric(round.(H[1:i, 1:i], 13))
    end
end