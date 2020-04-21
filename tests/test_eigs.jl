@testset "eigvals                                " begin
    # see Trefthen and Bau for this test case
    Random.seed!(1)
    m = 200
    A = 2*Matrix{Float64}(I, m, m) + 0.5*randn(m, m)/sqrt(m)
    b = ones(m)
        
    # solve
    evals = GMRES.eigvals(A, b, 4, 100)

    # test output
    @test length(evals[1]) == 100
    @test length(evals[2]) == 100
    @test length(evals[3]) == 100
    @test length(evals[4]) == 100

    # test convergence
    real_eigvals = sort(LinearAlgebra.eigvals(A); by=real, rev=true)
    for i = 1:4
        @test abs(real(evals[i][end]) - real(real_eigvals[i])) < 1e-7
        @test abs(imag(evals[i][end]) - imag(real_eigvals[i])) < 1e-7
    end
end