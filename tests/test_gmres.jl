@testset "trefthen                               " begin
    # see Trefthen and Bau for this test case
    Random.seed!(1)
    m = 200
    A = 2*Matrix{Float64}(I, m, m) + 0.5*randn(m, m)/sqrt(m)
    b = randn(m)
        
    # solve
    x, res_norm = gmres!(A, deepcopy(b), 1e-7, 10, false)

    # check convergence rate
    expected = [4.0^(-n) for n = 1:10]
    @test res_norm < 1.1*(4.0^(-10.0))
    @test res_norm > 0.7*(4.0^(-10.0))
end

@testset "norm of error                          " begin
    # see Trefthen and Bau for this test case
    Random.seed!(1)
    m = 200
    A = 2*Matrix{Float64}(I, m, m) + 0.5*randn(m, m)/sqrt(m)
    x_ex = randn(m)
    b = A*x_ex
        
    # solve with large number of iterations
    x, res = gmres!(A, deepcopy(b), 1e-7, 20, false)

    # norm of error
    @test norm(x - x_ex, 2)/norm(b) < 1e-7
    @test norm(A*x - b,  2)/norm(b) < 1e-7
end