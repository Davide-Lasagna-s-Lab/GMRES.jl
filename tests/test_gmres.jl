using GMRES
using Base.Test

@testset "gmres         " begin
    # see Trefthen and Bau for this test case
    m = 2000
    A = 2*eye(m) + 0.5*randn(m, m)/sqrt(m)
    b = randn(m)
        
    # solve
    x, convres = gmres!(A, b; tol=1e-7, maxiter=10)

    # check
    expected = [4.0^(-n) for n = 0:10]
    @test all(convres.hist .< expected*1.1)
    @test all(convres.hist .> expected*0.7)
end